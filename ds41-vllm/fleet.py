#!/usr/bin/env python3
"""Run on Spark 1. Standard-library SSH/Docker launcher for four Spark nodes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import time
import urllib.request

HERE = Path(__file__).resolve().parent
NAME = 'ds41-jj'
MODEL = 'DeepSeek-V4.1-Flash'


def load_config(path):
    c = json.loads(Path(path).read_text())
    assert len(c['nodes']) == 4
    assert len({n['host'] for n in c['nodes']}) == 4
    assert len({n['ip'] for n in c['nodes']}) == 4
    assert len(c['hcas']) == 2
    assert c['max_num_seqs'] == 8, 'This is the c8 recipe'
    assert 0 < c['gpu_memory_utilization'] < 1
    assert 0 < c['max_model_len'] <= 1048576
    assert c['draft_tokens'] in (0, 1, 3, 5, 7)
    for key in ('model_path', 'cache_path'):
        assert c[key].startswith('/') and c[key] != '/' and ',' not in c[key]
    return c


def ssh(c, rank):
    args = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
    if c.get('ssh_identity'):
        args += ['-i', c['ssh_identity']]
    return args + [f"{c['ssh_user']}@{c['nodes'][rank]['host']}"]


def remote(c, rank, script, timeout=120):
    proc = subprocess.run(ssh(c, rank) + ['bash', '-s'], input='set -euo pipefail\n' + script,
                          text=True, capture_output=True, timeout=timeout)
    if proc.returncode:
        raise RuntimeError(f'rank {rank}: {proc.stdout}\n{proc.stderr}')
    return proc.stdout.strip()


def environment(c, rank):
    return {
        'VLLM_HOST_IP': c['nodes'][rank]['ip'],
        'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'VLLM_USE_V2_MODEL_RUNNER': '1',
        'VLLM_ENABLE_ROCE_ALLREDUCE': '1', 'VLLM_ENABLE_PCIE_ALLREDUCE': '0',
        'VLLM_ALLREDUCE_USE_SYMM_MEM': '0', 'VLLM_ALLREDUCE_USE_FLASHINFER': '0',
        'VLLM_ROCE_ALLREDUCE_MAX_SIZE': '2MB', 'VLLM_ROCE_ALLGATHER_MAX_SIZE': '16MB',
        'B12X_ROCE_HCA': ','.join(c['hcas']), 'B12X_ROCE_GID_INDEX': str(c['gid_index']),
        'NCCL_NET': 'IB', 'NCCL_IB_DISABLE': '0', 'NCCL_DEBUG': 'INFO',
        'NCCL_IB_HCA': '=' + ','.join(c['hcas']), 'NCCL_IB_GID_INDEX': str(c['gid_index']),
        'NCCL_NVLS_ENABLE': '0', 'NCCL_IB_MERGE_NICS': '0', 'NCCL_CROSS_NIC': '1',
        'NCCL_MIN_NCHANNELS': '4', 'NCCL_MAX_NCHANNELS': '4',
        'NCCL_IGNORE_CPU_AFFINITY': '1',
        'CUTE_DSL_ARCH': 'sm_121a', 'TORCH_CUDA_ARCH_LIST': '12.1a',
        'CUDA_DEVICE_MAX_CONNECTIONS': '32', 'OMP_NUM_THREADS': '8',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
        'MALLOC_ARENA_MAX': '2', 'TOKENIZERS_PARALLELISM': 'false',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'XDG_CACHE_HOME': '/cache', 'VLLM_USE_BREAKABLE_CUDAGRAPH': '0',
    }


def docker(c, rank, name=None):
    cmd = ['docker', 'run', '--gpus', 'all', '--network', 'host', '--ipc', 'host',
           '--ulimit', 'memlock=-1:-1', '--ulimit', 'nofile=1048576:1048576',
           '--device', '/dev/infiniband:/dev/infiniband',
           '--security-opt', 'seccomp=unconfined',  # io_uring is blocked by Docker's default profile
           '--mount', f"type=bind,src={c['model_path']},dst=/checkpoint,readonly",
           '--mount', f"type=bind,src={c['cache_path']},dst=/cache",
           '--entrypoint', '', '-e', 'NCCL_SOCKET_IFNAME', '-e', 'GLOO_SOCKET_IFNAME']
    cmd += ['--detach', '--name', name] if name else ['--rm']
    for key, value in environment(c, rank).items():
        cmd += ['-e', f'{key}={value}']
    return cmd + [c['image']]


def setup(c, rank):
    # The interface is discovered separately on every node, never copied from rank 0.
    ip = shlex.quote(c['nodes'][rank]['ip'])
    return f'''iface=$(ip -o -4 address show | awk -v wanted={ip} 'split($4,a,"/") && a[1]==wanted {{print $2}}')
test -n "$iface"
test "$(printf '%s\\n' "$iface" | wc -l)" = 1
export NCCL_SOCKET_IFNAME="=$iface" GLOO_SOCKET_IFNAME="$iface"
'''


def serve_args(c, rank):
    cmd = ['vllm', 'serve', f"/checkpoint/snapshots/{c['revision']}",
           '--served-model-name', MODEL, '--host', '0.0.0.0', '--port', str(c['port']),
           '--distributed-executor-backend', 'mp', '--nnodes', '4', '--node-rank', str(rank),
           '--master-addr', c['nodes'][0]['ip'], '--master-port', str(c['master_port']),
           '--tensor-parallel-size', '4', '--decode-context-parallel-size', '1',
           '--dtype', 'bfloat16', '--load-format', 'safetensors', '--safetensors-load-strategy', 'lazy',
           '--attention-backend', 'B12X', '--linear-backend', 'b12x', '--moe-backend', 'b12x',
           '--block-size', '256', '--kv-cache-dtype', 'fp8',
           '--engram-config', json.dumps({'cpu_offload': False, 'table_memory': 'disk'}),
           '--gpu-memory-utilization', str(c['gpu_memory_utilization']),
           '--max-model-len', str(c['max_model_len']), '--max-num-seqs', '8',
           '--max-num-batched-tokens', str(c['max_num_batched_tokens']),
           '--enable-prefix-caching', '--enable-chunked-prefill', '--async-scheduling',
           '--no-scheduler-reserve-full-isl',
           '--generation-config', 'vllm', '--reasoning-parser', 'deepseek_v41',
           '--tool-call-parser', 'deepseek_v41', '--enable-auto-tool-choice']
    depth = c['draft_tokens'] + 1
    compilation = {'cudagraph_mode': 'FULL_AND_PIECEWISE', 'custom_ops': ['all'],
                   'cudagraph_capture_sizes': list(range(1, 8 * depth + 1)),
                   'pass_config': {'fuse_allreduce_rms': False}}
    cmd += ['--compilation-config', json.dumps(compilation)]
    if c['draft_tokens']:
        cmd += ['--speculative-config', json.dumps({
            'method': 'dspark', 'num_speculative_tokens': c['draft_tokens'],
            'draft_tensor_parallel_size': 4, 'attention_backend': 'B12X',
            'draft_sample_method': 'greedy', 'rejection_sample_method': 'standard',
            'enable_adaptive_verification': True})]
    if rank:
        cmd += ['--headless']
    return cmd


def plan(c, rank):
    return setup(c, rank) + shlex.join(docker(c, rank, f'{NAME}-{rank}') + serve_args(c, rank)) + '\n'


def preflight(c):
    ids = []
    for rank in range(4):
        script = setup(c, rank) + 'test "$(uname -m)" = aarch64\n'
        script += shlex.join(['mkdir', '-p', c['cache_path']]) + '\n'
        script += shlex.join(['test', '-r', f"{c['model_path']}/snapshots/{c['revision']}/config.json"]) + '\n'
        script += shlex.join(['docker', 'image', 'inspect', '--format', '{{.Os}}/{{.Architecture}}', c['image']]) + " | grep -qx linux/arm64\n"
        script += f"fstype=$(findmnt -n -o FSTYPE -T {shlex.quote(c['model_path'])})\n"
        script += 'case "$fstype" in ext4|xfs|btrfs) ;; *) echo "Model must be local SSD storage, got $fstype"; exit 1;; esac\n'
        for hca in c['hcas']:
            base = f'/sys/class/infiniband/{hca}/ports/1'
            script += shlex.join(['grep', '-q', 'ACTIVE', base + '/state']) + '\n'
            script += shlex.join(['grep', '-q', 'RoCE v2', base + f"/gid_attrs/types/{c['gid_index']}"]) + '\n'
        check = '''import ctypes
lib=ctypes.CDLL('liburing.so.2')
ring=ctypes.create_string_buffer(4096)
rc=lib.io_uring_queue_init(8,ring,0)
assert rc==0, f'io_uring denied/unavailable: {rc}'
lib.io_uring_queue_exit(ring)
from b12x.comm import roce
assert roce.is_supported(), 'RoCEnante unsupported'
'''
        script += shlex.join(docker(c, rank) + ['python3', '-c', check]) + '\n'
        script += shlex.join(docker(c, rank) + ['python3', '/opt/ds41/image-check.py', '--gpu']) + '\n'
        script += shlex.join(docker(c, rank) + ['python3', '/opt/ds41/model-check.py', f"/checkpoint/snapshots/{c['revision']}"]) + '\n'
        script += shlex.join(['docker', 'image', 'inspect', '--format', '{{.Id}}', c['image']]) + '\n'
        result = remote(c, rank, script, timeout=600)
        print(f'rank {rank}: {result}', flush=True)
        ids.append(result.splitlines()[-1])
    assert len(set(ids)) == 1, f'Image IDs differ: {ids}'


def fabric(c):
    names = [f'{NAME}-fabric-{rank}' for rank in range(4)]
    launched = []
    try:
        for rank in reversed(range(4)):
            cmd = docker(c, rank, names[rank]) + [
                'torchrun', '--nnodes=4', '--nproc-per-node=1', f'--node-rank={rank}',
                f"--master-addr={c['nodes'][0]['ip']}", f"--master-port={c['master_port'] + 1}",
                '/opt/ds41/roce-check.py']
            remote(c, rank, setup(c, rank) + shlex.join(cmd))
            launched.append(rank)
        def wait(rank):
            return remote(c, rank, shlex.join(['docker', 'wait', names[rank]]), timeout=900)
        with ThreadPoolExecutor(max_workers=4) as pool:
            codes = list(pool.map(wait, range(4)))
        for rank in range(4):
            output = remote(c, rank, shlex.join(['docker', 'logs', names[rank]]) + ' 2>&1')
            print(f'fabric rank {rank}: {output}', flush=True)
            assert codes[rank] == '0' and '"status": "PASS"' in output, 'Fabric qualification failed'
    finally:
        for rank in launched:
            remote(c, rank, shlex.join(['docker', 'rm', '-f', names[rank]]))


def request(c, path, payload=None):
    url = f"http://{c['nodes'][0]['ip']}:{c['port']}{path}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode() if payload is not None else None,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        body = response.read()
        return json.loads(body) if body else None


def smoke(c):
    import math
    def generate(i):
        result = request(c, '/v1/completions', {
            'model': MODEL, 'prompt': f'Question {i}: The four inner planets are',
            'max_tokens': 48, 'temperature': 0, 'logprobs': 1})
        choice = result['choices'][0]
        assert choice['text'].strip(), result
        probs = choice['logprobs']['token_logprobs']
        assert probs and all(p is not None and math.isfinite(p) for p in probs), result
        return result['usage']['completion_tokens']
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as pool:
        counts = list(pool.map(generate, range(8)))
    print(json.dumps({'concurrency': 8, 'completion_tokens': sum(counts),
                      'wall_seconds': time.monotonic() - start}))
    chat = request(c, '/v1/chat/completions', {
        'model': MODEL, 'messages': [{'role': 'user', 'content': 'Name the four inner planets.'}],
        'max_tokens': 128, 'temperature': 0, 'chat_template_kwargs': {'thinking': False}})
    assert chat['choices'][0]['message'].get('content'), chat
    print('Chat smoke:', chat['choices'][0]['message']['content'])
    for rank in range(4):
        state = remote(c, rank, shlex.join(['docker', 'inspect', '--format', '{{json .State}}', f'{NAME}-{rank}']))
        state = json.loads(state)
        assert state['Running'] and not state['OOMKilled'], (rank, state)
    output = remote(c, 0, shlex.join(['docker', 'logs', f'{NAME}-0']) + ' 2>&1')
    assert 'Using RoCEnante (b12x one-shot RoCE collectives)' in output, 'RoCEnante initialization unconfirmed'
    assert 'RoCEnante all-reduce is live' in output, 'No confirmed RoCEnante dispatch'
    print('c8 smoke and live RoCEnante dispatch passed; inspect generated output for quality.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(HERE / 'cluster-c8.json'))
    parser.add_argument('action', choices=['plan', 'share', 'preflight', 'fabric', 'start', 'smoke', 'status', 'logs', 'stop'])
    parser.add_argument('--rank', type=int, choices=range(4), default=0)
    args = parser.parse_args()
    c = load_config(args.config)
    if args.action == 'plan':
        for rank in range(4):
            print(f'# rank {rank} on {c["nodes"][rank]["host"]}\n{plan(c, rank)}')
    elif args.action == 'share':
        image_id = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', c['image']], text=True).strip()
        for rank in range(4):
            current = remote(c, rank, shlex.join(['docker', 'image', 'inspect', '--format', '{{.Id}}', c['image']]) + ' 2>/dev/null || true')
            if current == image_id:
                print(f'rank {rank}: already has {image_id}')
                continue
            save = subprocess.Popen(['docker', 'save', c['image']], stdout=subprocess.PIPE)
            try:
                subprocess.run(ssh(c, rank) + ['docker', 'load'], stdin=save.stdout, check=True)
            finally:
                save.stdout.close()
            assert save.wait() == 0
            actual = remote(c, rank, shlex.join(['docker', 'image', 'inspect', '--format', '{{.Id}}', c['image']]))
            assert actual == image_id, (rank, actual, image_id)
            print(f'rank {rank}: {actual}')
    elif args.action == 'preflight':
        preflight(c)
    elif args.action == 'fabric':
        preflight(c)
        fabric(c)
    elif args.action == 'start':
        # Refuse duplicate deployment before probing or allocating model memory.
        for rank in range(4):
            remote(c, rank, f'! docker container inspect {NAME}-{rank} >/dev/null 2>&1')
        preflight(c)
        fabric(c)
        for rank in reversed(range(4)):
            print(remote(c, rank, plan(c, rank)), flush=True)
        deadline = time.monotonic() + c['startup_timeout']
        while True:
            try:
                request(c, '/health')
                break
            except (OSError, ValueError):
                if time.monotonic() > deadline:
                    raise TimeoutError('Startup timed out; retain containers for logs, then run stop.')
                for rank in range(4):
                    state = remote(c, rank, f"docker inspect --format '{{{{.State.Running}}}}' {NAME}-{rank}")
                    assert state == 'true', f'Rank {rank} exited; run logs --rank {rank}'
                print('Waiting for model startup...', flush=True)
                time.sleep(10)
        smoke(c)
        print(f"Ready: http://{c['nodes'][0]['ip']}:{c['port']}/v1")
    elif args.action == 'smoke':
        smoke(c)
    elif args.action == 'logs':
        subprocess.run(ssh(c, args.rank) + ['docker', 'logs', '--tail', '200', f'{NAME}-{args.rank}'], check=True)
    else:
        for rank in range(4):
            if args.action == 'stop':
                script = f'if docker container inspect {NAME}-{rank} >/dev/null 2>&1; then docker rm -f {NAME}-{rank}; fi'
            else:
                script = shlex.join(['docker', 'ps', '-a', '--filter', f'name=^/{NAME}-{rank}$'])
            print(f'rank {rank}: {remote(c, rank, script)}')


if __name__ == '__main__':
    main()
