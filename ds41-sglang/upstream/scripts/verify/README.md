# Verification scripts (2026-09-11, REPORT.md section 17)

Run the Python scripts from a worker (spark2), never on the head; `memguard.py` and the
ramp drivers run on the head.

- `memguard.py <log> [threshold_gb]` (head): samples MemAvailable every second, POSTs
  `/abort_request` below the threshold. Arm it for every long-prompt test; stop it after.
- `batchtest.py <tag>`: greedy batches of 3 and 4 plus a sampled batch of 4 (prose, 120
  tokens); flags symbol/CJK garbage. The NaN-prefill fault of section 17 shows here.
- `diag5.py`: virgin-engine probe (600- and 150-token prompts, batches of 3 and 4), then a
  replay of the boot.py warm-up pieces with a probe after each, then a length sweep.
- `prefill_distinct.py <tokens> <seed>`: one needle prompt of ~tokens with seeded content
  (no radix hits). `ramp2.sh <tokens> <seed...>` (head) runs several under the guard and
  reports MemAvailable before / lowest / after.
- `throughput.py`: greedy and sampled throughput at concurrency 1-4 (usage-based).
- `idle_drift.sh` (head): MemAvailable every 30 s for 10 minutes.

Raw outputs of the 2026-09-11 boots: `logs/profile-2026-09-10/boots-2026-09-11/`.
