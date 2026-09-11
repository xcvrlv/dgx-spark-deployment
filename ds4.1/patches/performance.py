#!/usr/bin/env python3
"""Hash-gated optional sector sizing and sampled native disk timing."""
import argparse
import hashlib
import json
from pathlib import Path


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one performance anchor: {old[:80]!r}")
    return text.replace(old, new)


def reader_patch(text):
    text = replace(text, "    unsigned slots_count;", "    unsigned slots_count, block_bytes;")
    anchor = "    /* At most ceil((row_bytes + block - 1) / block) fragments per plane. */"
    text = replace(text, anchor, '''    const char *setting = getenv("DS41_DISK_BLOCK_BYTES");
    unsigned block_bytes = PLE_BLOCK;
    if (setting && strcmp(setting, "512") == 0) block_bytes = 512u;
    else if (setting && strcmp(setting, "4096") != 0)
        return PyErr_Format(PyExc_ValueError, "DS41_DISK_BLOCK_BYTES must be 512 or 4096");
''' + anchor)
    text = text.replace("(size_t)weight_bytes / PLE_BLOCK", "(size_t)weight_bytes / block_bytes")
    text = text.replace("(size_t)scale_bytes / PLE_BLOCK", "(size_t)scale_bytes / block_bytes")
    text = replace(text, "    reader->shard_rows = shard_rows;", "    reader->block_bytes = block_bytes;\n    reader->shard_rows = shard_rows;")
    # Keep 4096-byte buffer alignment; only request offsets and lengths change.
    text = replace(text, "unsigned length = PLE_BLOCK - (uint64_t)offset % PLE_BLOCK;",
                   "unsigned length = reader->block_bytes - (uint64_t)offset % reader->block_bytes;")
    text = text.replace("(int64_t)(PLE_BLOCK - 1)", "(int64_t)(reader->block_bytes - 1)")
    text = text.replace("block - last != PLE_BLOCK", "block - last != reader->block_bytes")
    text = text.replace("(unsigned)(last - first) + PLE_BLOCK", "(unsigned)(last - first) + reader->block_bytes")
    return text


def timing_patch(text):
    text = replace(text, "import threading", "import threading\nimport time\nimport json")
    text = replace(text, "        self._cache_used = False", '''        self._cache_used = False
        self._ds41_every = int(os.environ.get("DS41_DISK_LOG_EVERY", "0"))
        self._ds41_totals = dict(calls=0, wall_seconds=0., execution_seconds=0.,
                                 requested_bytes=0, read_bytes=0, read_calls=0)
'''.rstrip())
    text = replace(text, "        self.ids_host[:count].copy_(ids.view(-1)[:count], non_blocking=True)",
                   "        started = time.perf_counter() if self._ds41_every else 0.\n        self.ids_host[:count].copy_(ids.view(-1)[:count], non_blocking=True)")
    text = replace(text, "\n    def stats(self) -> dict[str, int | float]:", '''
        if self._ds41_every:
            wall = time.perf_counter() - started
            stats = self.stats()
            totals = self._ds41_totals
            totals["calls"] += 1
            totals["wall_seconds"] += wall
            for key in ("execution_seconds", "requested_bytes", "read_bytes", "read_calls"):
                totals[key] += stats[key]
            if totals["calls"] % self._ds41_every == 0:
                print("DS41_DISK " + json.dumps(dict(totals, pid=os.getpid(),
                    table_rows=self.table_rows, last_lookups=count,
                    block_bytes=int(os.environ.get("DS41_DISK_BLOCK_BYTES", "4096")),
                    owned_staging_bytes=stats["owned_staging_bytes"])), flush=True)

    def stats(self) -> dict[str, int | float]:''')
    return text


TRANSFORMS = {"b12x/loader/_ple_reader.c": reader_patch,
              "b12x/sequence/_shared/disk_table.py": timing_patch}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name("performance-hashes.json").read_text())
    pending = []
    for relative, transform in TRANSFORMS.items():
        path = args.root / relative
        text = path.read_text().rstrip() + "\n"
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest == manifest[relative]["output"]:
            continue
        assert digest == manifest[relative]["input"], f"Unexpected source: {path}"
        patched = transform(text)
        assert hashlib.sha256(patched.encode()).hexdigest() == manifest[relative]["output"]
        pending.append((path,patched))
    for path, patched in pending:
        path.write_text(patched)
    print("DS41 performance patch verified/applied: disk-sector-v1")


if __name__ == "__main__":
    main()
