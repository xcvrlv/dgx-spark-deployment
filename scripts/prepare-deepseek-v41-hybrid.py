#!/usr/bin/env python3
"""Prepare pinned MXFP4 + FP4 Engram weights on Linux; optionally rsync peers.

This builds a checkpoint, not an inference-engine patch. See the companion doc.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import struct
import subprocess

UPSTREAM = "deepseek-ai/DeepSeek-V4.1-Flash"
UP_REV = "dba1be0a40aa45a94ad051997016db3960a90277"
ENGRAM = "LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4"
ENG_REV = "dfce15b92ed1fa76e80e2a46ba847e5b5451f12c"
SHARDS = [f"model-{i:05d}-of-00048.safetensors" for i in range(1, 49)]
ROOT = Path(__file__).resolve().parent.parent


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def header(path):
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if not 2 <= size <= 64 * 1024 * 1024:
            raise ValueError(f"Invalid safetensors header: {path}")
        tensors = json.loads(stream.read(size))
    tensors.pop("__metadata__", None)
    end = 0
    for entry in sorted(tensors.values(), key=lambda t: t["data_offsets"]):
        start, stop = entry["data_offsets"]
        if start != end or stop < start:
            raise ValueError(f"Invalid tensor offsets: {path}")
        end = stop
    if 8 + size + end != path.stat().st_size:
        raise ValueError(f"Truncated or oversized shard: {path}")
    return tensors, end


def build_index(directory, expected):
    mapping, total = {}, 0
    for name in SHARDS:
        tensors, size = header(directory / name)
        total += size
        for key, tensor in tensors.items():
            if key in mapping or expected.get(key) != name:
                raise ValueError(f"Unexpected/duplicate/misplaced tensor: {key}")
            mapping[key] = name
            if key in ("layers.1.engram.embed.weight", "layers.14.engram.embed.weight"):
                if tensor["dtype"] != "U8" or tensor["shape"][-1] != 128:
                    raise ValueError(f"Expected packed FP4 Engram: {key}")
            if key in ("layers.1.engram.embed.scale", "layers.14.engram.embed.scale"):
                if tensor["dtype"] != "F8_E8M0" or tensor["shape"][-1] != 8:
                    raise ValueError(f"Expected block-32 E8M0 Engram scale: {key}")
    if mapping != expected:
        raise ValueError("Hybrid tensor set differs from upstream")
    return {"metadata": {"total_size": total}, "weight_map": mapping}


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def peers(args):
    if args.hosts:
        hosts = args.hosts.split(",")
    else:
        inventory = dict(re.findall(r"^(SPARK_\d+_CX0_IP)=(\S+)$",
                                    args.inventory.read_text(), re.MULTILINE))
        hosts = [inventory[f"SPARK_{i}_CX0_IP"] for i in range(2, 5)]
    targets = [f"{args.user}@{host}" if args.user else host for host in hosts]
    for target in targets:
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", target):
            raise ValueError(f"Invalid SSH target: {target}")
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="New dedicated directory, same absolute path on every node")
    parser.add_argument("--copy", action="store_true", help="Copy to Sparks 2-4 after verification")
    parser.add_argument("--hosts", help="Comma-separated peer hosts; defaults to inventory Sparks 2-4")
    parser.add_argument("--user", default=os.environ.get("SPARK_SSH_USER", ""))
    parser.add_argument("--ssh-key", default=os.environ.get("SPARK_SSH_KEY", ""))
    parser.add_argument("--inventory", type=Path, default=ROOT / "sparks.env")
    parser.add_argument("--plan", action="store_true", help="Print selection without network or writes")
    args = parser.parse_args()
    directory = args.model_dir.expanduser().resolve()
    targets = peers(args) if args.copy else []
    print(f"Upstream {UP_REV}: shards 1-46; Engram {ENG_REV}: shards 47-48", flush=True)
    print(f"Destination: {directory}; ~383.7 GiB weights per host; peers: {targets}", flush=True)
    if args.plan:
        return
    if os.name != "posix":
        raise RuntimeError("Run this script on the Linux head Spark")
    from huggingface_hub import HfApi, hf_hub_download
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / ".prepare.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    provenance = {"upstream": UPSTREAM, "upstream_revision": UP_REV,
                  "engram": ENGRAM, "engram_revision": ENG_REV}
    marker = directory / ".hybrid-source.json"
    if marker.exists():
        if json.loads(marker.read_text()) != provenance:
            raise RuntimeError("Directory belongs to a different build")
    elif any(p.name != ".prepare.lock" for p in directory.iterdir()):
        raise RuntimeError("Use a new dedicated directory, not an existing model or HF snapshot")
    write_json(marker, provenance)
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    if args.ssh_key:
        ssh += ["-i", str(Path(args.ssh_key).expanduser())]
    # Preflight before committing to the large download. Never clear an existing model.
    for target in targets:
        check = "import pathlib,sys,shutil; p=pathlib.Path(sys.argv[1]); " \
                "p.mkdir(parents=True,exist_ok=True); " \
                "assert not any(p.iterdir()) or (p/'.hybrid-source.json').read_text()==sys.argv[2], " \
                "'Destination is not this hybrid build'; " \
                "print('Free GiB:',shutil.disk_usage(p).free/2**30)"
        subprocess.run(ssh + [target, shlex.join(["python3", "-c", check, str(directory),
                                                 marker.read_text()])], check=True)
        subprocess.run(ssh + [target, "command -v rsync && command -v sha256sum"], check=True)
    if targets and shutil.which("rsync") is None:
        raise RuntimeError("Install rsync on the head Spark")

    api = HfApi()
    info = api.model_info(UPSTREAM, revision=UP_REV, files_metadata=True)
    eng_info = api.model_info(ENGRAM, revision=ENG_REV, files_metadata=True)
    up_files = {f.rfilename: f for f in info.siblings}
    eng_files = {f.rfilename: f for f in eng_info.siblings}
    # Keep source config/index separate so repeat downloads never overwrite hybrid metadata.
    source = directory / "provenance"
    source.mkdir(exist_ok=True)
    for name in ("config.json", "model.safetensors.index.json"):
        hf_hub_download(UPSTREAM, name, revision=UP_REV, local_dir=source)
    expected = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    if set(expected.values()) != set(SHARDS):
        raise RuntimeError("Pinned upstream shard layout changed unexpectedly")
    selected = [(UPSTREAM, UP_REV, up_files[n]) for n in SHARDS[:46]]
    selected += [(ENGRAM, ENG_REV, eng_files[n]) for n in SHARDS[46:]]
    extras = [n for n in up_files if not n.endswith(".safetensors") and
              n not in ("config.json", "model.safetensors.index.json", ".gitattributes")]
    selected += [(UPSTREAM, UP_REV, up_files[n]) for n in extras]
    needed = sum(f.size for _, _, f in selected if not (directory / f.rfilename).exists())
    if shutil.disk_usage(directory).free < needed + 5 * 2**30:
        raise RuntimeError(f"Need at least {needed / 2**30 + 5:.1f} GiB free for remaining files")
    for target in targets:
        capacity = "import pathlib,sys,json,shutil; p=pathlib.Path(sys.argv[1]); " \
                   "files=json.loads(sys.argv[2]); " \
                   "need=sum(s for n,s in files if not (p/n).exists() or (p/n).stat().st_size!=s); " \
                   "assert shutil.disk_usage(p).free >= need+5*2**30, 'Insufficient peer disk space'"
        subprocess.run(ssh + [target, shlex.join([
            "python3", "-c", capacity, str(directory),
            json.dumps([(f.rfilename, f.size) for _, _, f in selected])])], check=True)
    (directory / "HYBRID_READY.json").unlink(missing_ok=True)
    hashes = {}
    for repo, revision, entry in selected:
        name = entry.rfilename
        print(f"Downloading/verifying {repo}: {name}", flush=True)
        path = Path(hf_hub_download(repo, name, revision=revision, local_dir=directory))
        digest = checksum(path)
        if entry.lfs and digest != entry.lfs.sha256:
            raise RuntimeError(f"Source SHA256 mismatch: {path}; remove this file and rerun")
        hashes[name] = digest

    index = build_index(directory, expected)
    config = json.loads((source / "config.json").read_text())
    if config["quantization_config"]["expert_dtype"] != "fp4":
        raise RuntimeError("Expected upstream MXFP4 expert configuration")
    config["quantization_config"].update(engram_dtype="fp4", engram_block_size=32,
                                         engram_scale_fmt="ue8m0")
    write_json(directory / "config.json", config)
    write_json(directory / "model.safetensors.index.json", index)
    write_json(directory / "HYBRID_READY.json", {
        **provenance, "tensor_bytes": index["metadata"]["total_size"],
        "runtime_validated": False,
        "requires": "Packed FP4 Engram gather/dequant and distributed table sharding support",
    })
    for name in ("config.json", "model.safetensors.index.json", "HYBRID_READY.json"):
        hashes[name] = checksum(directory / name)
    (directory / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n"
                                                  for name, digest in sorted(hashes.items())))
    for target in targets:
        print(f"Copying and verifying {target} (full checkpoint)", flush=True)
        # A failed or interrupted transfer must not advertise readiness.
        subprocess.run(ssh + [target, shlex.join(["rm", "-f", "--",
                                                 str(directory / "HYBRID_READY.json")])], check=True)
        subprocess.run(["rsync", "-a", "--checksum", "--protect-args", "--partial", "--info=progress2",
                        "--exclude=.cache/", "--exclude=.prepare.lock",
                        "--exclude=HYBRID_READY.json", "-e", shlex.join(ssh),
                        str(directory) + "/", f"{target}:{directory}/"], check=True)
        # Verify all payloads before publishing the readiness file.
        verify = f"cd {shlex.quote(str(directory))} && " \
                 "sed '/  HYBRID_READY.json$/d' SHA256SUMS | sha256sum -c -"
        subprocess.run(ssh + [target, verify], check=True)
        subprocess.run(["rsync", "-a", "--protect-args", "-e", shlex.join(ssh),
                        str(directory / "HYBRID_READY.json"), f"{target}:{directory}/"], check=True)
    print("Checkpoint prepared and requested copies verified. Engine compatibility is NOT validated.")


if __name__ == "__main__":
    main()
