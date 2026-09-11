#!/usr/bin/env python3
"""Head-Spark fleet launcher. All remote control and image transfer uses CX0 SSH."""
import argparse
import json
import os
import re
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
import urllib.request

from serve import command

HERE = Path(__file__).resolve().parent


def run(args, **kwargs):
    try:
        return subprocess.run(args, check=True, text=True, **kwargs)
    except subprocess.CalledProcessError as error:
        # Preserve captured preflight/build diagnostics when a child fails.
        if error.stdout:
            print(error.stdout,file=sys.stderr,flush=True)
        if error.stderr:
            print(error.stderr,file=sys.stderr,flush=True)
        raise


def environment(c, rank, iface):
    return {
        "DS41_CONFIG_JSON":json.dumps(c),
        "CUDA_VISIBLE_DEVICES":"0", "CUDA_HOME":"/usr/local/cuda",
        "TRITON_PTXAS_PATH":"/usr/local/cuda/bin/ptxas", "CUTE_DSL_ARCH":"sm_121a",
        "TORCH_CUDA_ARCH_LIST":"12.1a", "PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True",
        "VLLM_USE_V2_MODEL_RUNNER":"1", "VLLM_WORKER_MULTIPROC_METHOD":"spawn",
        "VLLM_USE_BREAKABLE_CUDAGRAPH":str(int(c.get("graph_mode") == "FULL_AND_PIECEWISE")),
        "DS41_DISK_BLOCK_BYTES":str(c.get("disk_block_bytes",4096)),
        "DS41_DISK_LOG_EVERY":str(c.get("disk_log_every",0)),
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS":"1",
        "VLLM_ENABLE_PCIE_ALLREDUCE":"0", "VLLM_ENABLE_ROCE_ALLREDUCE":"1",
        "VLLM_ROCE_ALLREDUCE_MAX_SIZE":"2MB", "VLLM_ROCE_ALLGATHER_MAX_SIZE":"16MB",
        "B12X_ROCE_HCA":",".join(c["hcas"]), "B12X_ROCE_GID_INDEX":str(c["gid_index"]),
        "B12X_ROCE_CACHE_DIR":"/cache/b12x-roce",
        "LD_PRELOAD":"/opt/sparkring/nccl/libnccl.so.2",
        "VLLM_NCCL_SO_PATH":"/opt/sparkring/nccl/libnccl.so.2",
        "NCCL_NET":"IB", "NCCL_IB_DISABLE":"0", "NCCL_NVLS_ENABLE":"0",
        "NCCL_IB_HCA":",".join(c["hcas"]), "NCCL_IB_GID_INDEX":str(c["gid_index"]),
        "NCCL_MIN_NCHANNELS":"4", "NCCL_MAX_NCHANNELS":"4", "NCCL_CROSS_NIC":"1",
        "NCCL_IB_MERGE_NICS":"0", "NCCL_IB_SUBNET_AWARE_ROUTING":"1",
        "NCCL_IGNORE_CPU_AFFINITY":"1", "NCCL_DEBUG":"INFO",
        "NCCL_SOCKET_IFNAME":iface, "GLOO_SOCKET_IFNAME":iface, "TP_SOCKET_IFNAME":iface,
        "VLLM_HOST_IP":c["nodes"][rank]["cx0"], "OMP_NUM_THREADS":"8",
        "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1", "HF_HOME":"/cache/huggingface",
        "XDG_CACHE_HOME":"/cache", "VLLM_CACHE_ROOT":"/cache/vllm",
        "TRITON_CACHE_DIR":"/cache/triton", "TOKENIZERS_PARALLELISM":"false",
        "MALLOC_ARENA_MAX":"2", "VLLM_ENGINE_READY_TIMEOUT_S":"7200",
        # Qualification logs must include actual capture, not just enabled flags.
        "VLLM_LOGGING_LEVEL":"DEBUG",
    }


def fabric(c, rank):
    addresses = json.loads(run(["ip","-j","-4","addr","show"],capture_output=True).stdout)
    owners = {a["local"]:item["ifname"] for item in addresses for a in item["addr_info"]}
    node = c["nodes"][rank]
    assert node["cx0"] in owners and node["cx1"] in owners, "Missing CX0/CX1 addresses"
    expected = {owners[node["cx0"]],owners[node["cx1"]]}
    found = set()
    for hca in c["hcas"]:
        port = Path("/sys/class/infiniband")/hca/"ports/1"
        gid = str(c["gid_index"])
        assert "ACTIVE" in (port/"state").read_text(), f"Inactive HCA: {hca}"
        netdev = (port/"gid_attrs/ndevs"/gid).read_text().strip()
        assert "v2" in (port/"gid_attrs/types"/gid).read_text(), f"Expected RoCEv2 GID: {hca}"
        found.add(netdev)
    assert found == expected, f"HCA/GID interface mismatch: {found} vs {expected}"
    return owners[node["cx0"]]


def docker_base(c, rank, iface):
    result = ["docker","run","--gpus","all","--network","host","--ipc","host",
              "--shm-size","32g","--ulimit","memlock=-1:-1",
              "--ulimit","nofile=1048576:1048576","--cap-add","IPC_LOCK",
              "--device","/dev/infiniband:/dev/infiniband",
              # Docker's default seccomp blocks io_uring. No --privileged needed.
              "--security-opt","seccomp=unconfined",
              "-v",f"{c['model_path']}:/model:ro", "-v",f"{c['cache_path']}:/cache",
              "-v",f"{HERE}:/opt/ds41:ro"]
    for key,value in environment(c,rank,iface).items():
        result += ["-e",f"{key}={value}"]
    return result


def inspect(c):
    return json.loads(run(["docker","inspect",c["container"]],capture_output=True).stdout)[0]


def node_action(c, action, rank):
    name = c["container"]
    if action == "stop":
        # Scoped to this recipe's container; no fleet-wide or GLM teardown.
        result = subprocess.run(["docker","inspect",name],capture_output=True)
        if result.returncode == 0:
            run(["docker","rm","-f",name])
        return
    if action == "logs":
        run(["docker","logs","--tail","150",name])
        return
    if action in ("status","verify"):
        info = inspect(c)
        state = info["State"]
        assert state["Running"] and not state["OOMKilled"] and info["RestartCount"] == 0, state
        print(f"rank={rank} running image={info['Image']}",flush=True)
        if action == "verify":
            logs = run(["docker","logs",name],stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout
            assert "CG Capture: mode=FULL," in logs, "No full CUDA graph capture logged"
            assert "Graph capturing finished" in logs, "No completed CUDA graph capture logged"
            if c.get("graph_mode") == "FULL_AND_PIECEWISE":
                assert "Captured breakable cudagraph" in logs, "No completed piecewise capture"
            else:
                assert "CG Capture: mode=PIECEWISE," not in logs, "Unexpected piecewise graph capture"
            assert "dspark" in logs.lower(), "No DSpark evidence in logs"
            assert "[DS41_FP4_DISK] format=fp4 stored_row_bytes=128" in logs, "No FP4 disk-table load evidence"
            print(f"rank={rank}: captured graphs and FP4 disk tables recorded",flush=True)
        return
    iface = fabric(c,rank)
    if action == "preflight":
        image = json.loads(run(["docker","image","inspect",c["image"]],capture_output=True).stdout)[0]
        assert image["Architecture"] == "arm64"
        labels = image["Config"]["Labels"]
        assert labels["local.ds41.vllm"] == c["vllm_commit"]
        assert labels["local.ds41.b12x"] == c["b12x_commit"]
        assert labels["local.ds41.fp4-engram"] == "ds41-fp4-disk-v2"
        if c.get("performance_patch"):
            assert labels.get("local.ds41.performance") == "disk-sector-v1", "Build the performance image first"
        assert Path(c["model_path"]).is_dir()
        Path(c["cache_path"]).mkdir(parents=True,exist_ok=True)
        base = docker_base(c,rank,iface)
        run(base+["--rm","--entrypoint","python3",c["image"],"/opt/ds41/preflight.py"])
        run(base+["--rm","--entrypoint","python3",c["image"],"/opt/ds41/gpu_smoke.py"])
        if c.get("performance_patch"):
            run(base+["--rm","--entrypoint","python3",c["image"],"/opt/ds41/disk-bench.py",
                      "--rank",str(rank),"--blocks",str(c.get("disk_block_bytes",4096)),
                      "--tokens","1","6","--repeats","1"])
        print(f"rank={rank} iface={iface} hcas={c['hcas']} image={image['Id']}",flush=True)
    elif action == "disk-bench":
        base = docker_base(c,rank,iface)
        run(base+["--rm","--entrypoint","python3",c["image"],"/opt/ds41/disk-bench.py","--rank",str(rank)])
    elif action == "start":
        # Preserve an existing service; user can explicitly stop/restart this one.
        assert subprocess.run(["docker","inspect",name],capture_output=True).returncode != 0, \
            "DS41 container already exists; inspect it or use cluster.py stop first"
        Path(c["cache_path"]).mkdir(parents=True,exist_ok=True)
        run(docker_base(c,rank,iface)+["-d","--init","--name",name,"--restart","no",
            "--entrypoint","python3",c["image"],"/opt/ds41/serve.py","--rank",str(rank)])
        print(f"rank={rank} launched on {iface}",flush=True)


def ssh(c, rank):
    host = c["nodes"][rank]["host"]
    target = f"{c['ssh_user']}@{host}" if c["ssh_user"] else host
    return ["ssh","-o","BatchMode=yes","-o","ConnectTimeout=15",target]


def remote(c, rank, action):
    if rank == 0:
        node_action(c,action,rank)
    else:
        cmd = ["python3",c["deploy_dir"]+"/cluster.py",action,"--node",str(rank)]
        run(ssh(c,rank)+[shlex.join(cmd)])


def sync(c, config_path):
    for rank in range(1,4):
        peer = ssh(c,rank)
        run(peer+[shlex.join(["mkdir","-p",c["deploy_dir"]])])
        run(["rsync","-a","--protect-args","--exclude=build/","--exclude=__pycache__/",
             "--exclude=results/", "-e",shlex.join(peer[:-1]), str(HERE)+"/",
             f"{peer[-1]}:{c['deploy_dir']}/"])
        run(["rsync","-a","--protect-args","-e",shlex.join(peer[:-1]),str(config_path),
             f"{peer[-1]}:{c['deploy_dir']}/cluster.json"])


def verify_http(c):
    url = f"http://127.0.0.1:{c['port']}/v1/chat/completions"
    # Two requests exercise replay after warmup. No full 1M-token allocation test.
    for _ in range(2):
        body = {"model":c["served_model_name"],"messages":[{"role":"user","content":"Say hello."}],
                "max_tokens":64,"temperature":0}
        req = urllib.request.Request(url,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req,timeout=300) as response:
            result = json.load(response)
        assert result.get("choices"),result
    print("HTTP generation smoke passed",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["plan","build","copy-image","sync","preflight","start","status","logs","stop","verify","disk-bench"])
    parser.add_argument("--config",type=Path,default=HERE/"cluster.json")
    parser.add_argument("--node",type=int,choices=range(4),help="Internal/single-node operation")
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    if args.action == "plan":
        for rank in range(4):
            print(f"rank={rank} CX0={c['nodes'][rank]['cx0']} HCA={c['hcas']}")
            print(shlex.join(command(c,rank)))
        return
    if args.node is not None:
        if args.action not in ("preflight","start","status","logs","stop","verify","disk-bench"):
            parser.error("--node is only valid for per-node operational actions")
        node_action(c,args.action,args.node)
        return
    if args.action == "build":
        run(["bash",str(HERE/("build-performance.sh" if c.get("performance_patch") else "build-image.sh"))])
    elif args.action == "copy-image":
        expected = run(["docker","image","inspect","--format","{{.Id}}",c["image"]],capture_output=True).stdout.strip()
        for rank in range(1,4):
            with subprocess.Popen(["docker","save",c["image"]],stdout=subprocess.PIPE) as export:
                run(ssh(c,rank)+["docker load"],stdin=export.stdout)
                export.stdout.close()
                assert export.wait() == 0
            actual = run(ssh(c,rank)+[shlex.join(["docker","image","inspect","--format","{{.Id}}",c["image"]])],capture_output=True).stdout.strip()
            assert actual == expected, "Image ID mismatch"
    elif args.action == "sync":
        sync(c,args.config)
    elif args.action == "disk-bench":
        assert c.get("performance_patch"), "Disk A/B requires the performance image"
        sync(c,args.config)
        for rank in range(4):
            remote(c,rank,"disk-bench")
    elif args.action in ("preflight","start"):
        if args.action == "start":
            with socket.socket() as sock:
                sock.bind(("0.0.0.0",c["port"]))
        sync(c,args.config)
        model_digests, image_ids = set(), set()
        for rank in range(4):
            if rank == 0:
                cmd = [sys.executable,str(HERE/"cluster.py"),"preflight",
                       "--config",str(args.config.resolve()),"--node","0"]
            else:
                cmd = ssh(c,rank)+[shlex.join(["python3",c["deploy_dir"]+"/cluster.py","preflight","--node",str(rank)])]
            output = run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout
            print(output,flush=True)
            model_digests.add(re.search(r"config_index_sha256=([0-9a-f]{64})",output).group(1))
            image_ids.add(re.search(r"image=(sha256:[0-9a-f]{64})",output).group(1))
        assert len(model_digests) == len(image_ids) == 1, "Models or image IDs differ across hosts"
        if args.action == "start":
            # Start workers before the head, as in the earlier GLM fleet launcher.
            for rank in (3,2,1,0):
                remote(c,rank,"start")
            deadline = time.monotonic()+c["ready_timeout_seconds"]
            while time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{c['port']}/health",timeout=5) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                for rank in range(4):
                    remote(c,rank,"status")
                print("Waiting for model load/graph capture...",flush=True)
                time.sleep(20)
            else:
                raise TimeoutError("Readiness timed out; containers retained for logs")
            verify_http(c)
            for rank in range(4):
                remote(c,rank,"verify")
    else:
        if args.action == "verify":
            verify_http(c)
        for rank in range(4):
            remote(c,rank,args.action)


if __name__ == "__main__":
    main()
