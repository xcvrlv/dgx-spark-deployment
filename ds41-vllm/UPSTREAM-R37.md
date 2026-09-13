# Latest JJ / requested R37 update

Live heads checked 2026-09-13 16:36 UTC:
- JJ: `9342b1ae80972a5e553c15793eb15982977a0097`
- b12x: `fd3c638c0891effb684efcc5df6e54ce5fd5d05e`

GitHub returned no tags or releases for this repository. R37 is the operator's
requested version name, not a verified tag. The image pins the actual current
`dev/jovian-judgement` head and b12x head, not a floating R37 alias.

JJ's head merges #743: owned attention results and bounded short index scans.
b12x's head merges #365: compact prefill tiles for the DS4.1 mHC projection.
The RoCEnante proxy/runtime and vLLM block_pool.py still match the previously
audited input hashes byte-for-byte. Their guarded local patches remain applicable.
The main image now includes the bounded-hash patch; no extra child build needed.
For an independent hash-off comparison, build with `PREFILL_HASHES=0 bash
build-image.sh`; that produces a separate `-hashes-off` image tag.

## DSpark7 capacity audit

Defaults are c16, max context 393216, GPU utilization 0.88, batch tokens 4096,
and draft_tokens 7. configure-c16.py explicitly sets depth7 instead of inheriting
an old value; --draft-tokens 0 or 5 selects a comparison profile.

fleet.py requests graph sizes 1 through 128 = 16*(7+1). Upstream selects actual
full/piecewise/drafter graph families; their progress-bar counts need not match.
The native attention planner reserves max(16*(1+2*7), graph cap) = 240 decode
rows for parallel-drafting profiling, bounded by the 4096 batch capacity.
Linear execution capacities include the batch capacity and graph/decode bounds.
CompressorStateCache uses max(8, 1 << (7+1).bit_length()) = 16 rows, allowing
drafts, bonus token and previous state to coexist. DSpark context graphs derive
capacities from the decode bound, backing hidden-state buffer and capture cap.
These are source-level checks, not proof that all buffers fit GPU memory.

## Startup failure in the supplied log

The failing _check_enough_kv_cache_memory branch tests available_memory <= 0.
get_kv_cache_configs checks each worker separately after subtracting the reserved
null-block pool cost. Rank0 reports 26.41 GiB; this alone cannot establish the
usable budgets on ranks1-3 or exclude a profiling/accounting problem. This error
is distinct from the later check for insufficient capacity for max_model_len.
The log explicitly uses utilization 0.8500, not our requested 0.88.
Graph capture completed, including DSpark. The shared-memory warning occurred
while compilation/capture was busy; it is not the fatal exception. Later NCCL
socket messages occur during shutdown. Do not bypass the cache memory check.

Save all four worker logs before stopping existing containers. Compare each
worker's Available KV cache memory and memory-profile lines. If needed inspect
other processes and host memory on each Spark; memory reclaimed after the
server exits is not evidence of the budget available during startup.

## Build and launch on the Spark

From the updated ds41-vllm directory:

```bash
(
set -euo pipefail
bash build-image.sh
python3 configure-c16.py --from-config .build/cluster-roce.json --output .build/cluster-c16-latest.json --draft-tokens 7
python3 fleet.py --config .build/cluster-c16-latest.json share
python3 fleet.py --config .build/cluster-c16-latest.json stop
python3 fleet.py --config .build/cluster-c16-latest.json start
)
```

No GPU image build or four-Spark run was performed from this Windows workspace.
Use the generated config consistently; this launch runs fabric and serving checks.
