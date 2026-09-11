#!/usr/bin/env python3
"""Collect complete DS41 container logs and state over CX0; never restart anything."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("cluster.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output or Path("ds41-logs-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    failed = False
    for rank, node in enumerate(config["nodes"]):
        operations = {
            "log": ["docker", "logs", "--timestamps", config["container"]],
            "state.txt": ["docker", "inspect", "--format", "{{json .State}}", config["container"]],
        }
        for suffix, command in operations.items():
            if rank:
                command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                           f"{config['ssh_user']}@{node['cx0']}", shlex.join(command)]
            target = output / f"rank-{rank}.{suffix}"
            with target.open("wb") as stream:
                try:
                    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
                    failed |= result.returncode != 0
                    print(f"{target}: exit={result.returncode}", flush=True)
                except (OSError, subprocess.TimeoutExpired) as error:
                    stream.write(f"\nCollection failed: {error}\n".encode())
                    failed = True
                    print(f"{target}: {error}", flush=True)
    print(f"Logs saved in {output.resolve()}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
