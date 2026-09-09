# EXL3 scheduling notes — safe image vs v10..v20, and kernel-path streamlining

Longer-horizon working notes, 2026-09-08. Companion to `CURRENT_WORK.md`; nothing
here changes serving defaults, image tags, or code. Facts are source-derived
from the pinned fixtures in `tmp/` unless explicitly marked **[user-observed]**
or **[estimate]**.

## 0. State decision recorded this session

- **[user-observed]** v20 serving ran with similar prefill and decode to the
  earlier SparkRing image + safe recipe comparison line. Decode may actually
  have improved, but the measurement included MTP3 draft/verify overhead, so
  the target-only decode rate was not isolated.
- **Decision: keep v20** (`recipes/glm53-exl3-v20-4x.yaml`,
  `VLLM_GB10_EXL3_FC1_TAILSPLIT=1`) as the R22-line serving baseline.
- **Follow-up (done):** target-only decode measured on both lines (2026-09-08,
  entries 1-2 in `TARGET-ONLY-PERFORMANCE.md`): safe prefill 741/757/711 at
  8k/64k/128k vs v20 645/645/628; decode ctx 0 (30-second samples) wins at
  conc 4 (+19.6%) and 8 (+5.1%), trails at conc 1 (-4.3%) and 2 (-7.2%). The
  keep-v20 decision rests on the MTP3 serving parity plus the
  production-concurrency decode wins; the clean tail-split isolation A/B is
  **in progress**: the `TAILSPLIT=0` arm is measured at 300-second decode
  (10.9/20.0/38.1/57.8 at conc 1/2/4/8, entry 3 in the log); the matched
  `TAILSPLIT=1` + safe 300-second reruns are pending. See
  `sparkrun-glm53-exl3/docs/r22-performance-v20.md` for scope and rollback.

## 1. Scheduling differences: safe (sparkring-switch-prefill-v2) vs v10..v20

The safe recipe predates versioning and runs a different image lineage
(SparkRing R7 vLLM at `/opt/venv`, NCCL at `/opt/sparkring/nccl`, plus exactly
one hash-gated indexer patch). It is not v9/v10 with old env names.

### 1.1 Identical scheduling knobs (verified in both recipes)

- `max_num_batched_tokens: 4096`, `max_num_seqs: 8`, `max_model_len: 1048576`,
  `--block-size 64`, `--enable-chunked-prefill`, `--enable-prefix-caching`,
  `--async-scheduling`, V2 model runner; both unset
  `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`.
- CUDA graphs: `FULL_DECODE_ONLY`, capture sizes [4,8,12,16,20,24,28,32],
  `custom_ops: all`, `fuse_allreduce_rms: false`; `VLLM_USE_BREAKABLE_CUDAGRAPH=0`;
  `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`. **Prefill is eager in both
  lines** (decode-only graphs).
- MTP: native MTP3, draft TP4, greedy sampling, spec-extend-as-decode, DCP
  global top-k + shard draft, instrumentation on.
- EXL3 plan env is byte-identical: `PREFILL_CAPACITY=4096`, `TRELLIS_MIN_M=1`,
  `TRELLIS_MAX_M=64`, `TRELLIS_BLOCK_M=8`, `PREFILL_TRELLIS=1`,
  `PREFILL_BLOCK_M=32`, `PREFILL_CHUNK=128` (chunk does not split the one-grid
  mixed path). Decode plan M8, prefill plan M32, unified tier dispatch.
- NCCL fabric env (IB, dual HCA, GID 3, 4 channels, subnet-aware routing).

⇒ The vLLM *core scheduler* config never differed. All differences live in the
attention/MoE execution paths, the transport selection, and the kernels.

### 1.2 Attention-path scheduling differences

| Item | Safe (R7 `B12X_MLA_SPARSE`) | v10..v20 (R22 `B12X`) |
| --- | --- | --- |
| Full-CKV gathered-prefill eligibility | no min-token gate | `CKV_GATHER_MIN_TOKENS=16` added (v10+); sub-16-token pure-prefill tails fall back to ordinary DCP where safe would gather |
| DCP query gather splitting | `VLLM_DCP_QUERY_SPLIT=1`, split above 8192 context tokens | absent; R22's NVIDIA GLM-DSA caller does not consume the R7 query-split settings (v10 tried re-adding them in v12 as "maybes", reverted in v13) |
| KV prefetch depth | explicitly `CKV_PREFETCH_DEPTH=0` | not set (R22 default) |
| DCP A2A | `VLLM_USE_B12X_DCP_A2A=1` (R7 path) | `VLLM_USE_DIRECT_DCP_A2A/Q_GATHER/KV_GATHER=0` trio + RoCEnante DCP adapter (v10+) |
| Indexer chunking | R7 generic splitter (logits-budget driven; 64 calls at 1M ctx) | v18 supertile coalescing (2 calls at 1M ctx) + v12-style fused CKV metadata + v16 bounded 256-row merge; safe has only the workspace rightsize patch |
| Causal-length metadata | per-layer multi-op PyTorch prep | v12 fused single-CTA metadata kernel; v18 removes `token_to_seq` map and device→host `.item()` per chunk |

At the 8k uncached prefill sample these attention-path differences collapse to:
min-token gate (tail chunks), no query split (R22), different metadata/prep
kernels, and transport routing. Since v20 measured similar prefill, none of
these is currently worth individual re-litigation; they matter only if we
chase long-context (>32k) prefill later, where indexer call counts diverge 64→2.

### 1.3 Transport scheduling timeline (matters when comparing any two versions)

- **Safe:** all NCCL for TP and DCP (no RoCEnante in the image).
- **v10..v12:** RoCEnante-eligible for TP + DCP small collectives (env on,
  no disabling flag).
- **v13..v18:** `--disable-custom-all-reduce` silently re-disabled RoCEnante
  (`_ENABLE_CUSTOM_ALL_REDUCE` gate) ⇒ **all-NCCL**, like safe. Any
  prefill/decode comparisons v13↔safe were NCCL-vs-NCCL;
  v10..v12 numbers were RoCEnante-in-the-loop.
- **v19+:** flag removed ⇒ RoCEnante live for TP all-reduce/all-gather
  (≤2 MiB / ≤16 MiB) and DCP gathers + ≤256 KiB reduce-scatter substitution;
  v15 inline payloads + balanced fanout; v16 `gather_into` + in-place CKV +
  skip-empty-CQ.
- Additional kernel-path deltas accumulated on the R22 line only:
  v11 fused bf16/fp16 final store (removes one conversion launch + ~192 MiB
  fp32 round-trip per layer at 4096 rows); v14 shared-input rotation reuse +
  compact input (7/8 fewer rotation butterflies/writes at prefill); v15 sigmoid
  SFU reciprocal + skinny (M≤2) dense profiles; v16 sigmoid compile fix;
  v19 FC2 group4 + M16; v20 FC1 ragged-wave tail split (decode).

### 1.4 Speculative-decoding scheduling differences

- Winner selection: safe and v10..v13 gather **full draft vocab logits** across
  TP (`use_local_argmax_reduction: false`); v14+ use **local argmax** +
  16-byte winner packets through RoCEnante direct gather
  (`VLLM_GB10_DRAFT_ARGMAX=1`). This is a real decode-path difference between
  safe and the current line — and one of the reasons a target-only decode
  comparison against old numbers is unreliable.
- `--no-enable-flashinfer-autotune`: present v10..v11 only (removed v12).
- Load format: safe `instanttensor` vs R22 `safetensors`; safe preloads compat
  libcuda; safe GPU memory utilization 0.91 vs 0.87 (v12+) — KV-pool/admission
  difference, not throughput scheduling.

### 1.5 Remaining unverified identity (open item)

The safe image's own `exl3.py` and `b12x` kernel sources have never been
hash-identified against the pinned fixtures used by the R22 line
(`kernel.py 591d06f2` = b12x `1e59a1f`; `mixed_trellis.py 4d5140de`;
`exl3.py 084d0305` = restored R7 layer + v13 dense-rotation fusion). Closing
this is one command on any Spark with the safe image loaded:

```bash
docker run --rm --entrypoint sha256sum spark-vllm-glm52-exl3:sparkring-switch-prefill-v2 \
  /opt/venv/lib/python3*/site-packages/b12x/moe/_shared/kernels/w4a16/kernel.py \
  /opt/venv/lib/python3*/site-packages/b12x/moe/_shared/kernels/w4a16/mixed_trellis.py \
  /opt/venv/lib/python3*/site-packages/vllm/model_executor/layers/quantization/exl3.py
```

If safe's `exl3.py` matches `084d0305` (pre-v13), the vLLM-side EXL3 plan
scheduling is confirmed shared lineage and the only EXL3 deltas vs safe are the
kernel overlays; if it drifts, diff it before trusting any MoE-path comparison.

## 2. EXL3 mixed-K kernel path: what actually runs (deep-dive)

Reconstructed exactly as shipped in v20 by applying the overlay chain to the
pinned fixtures: `kernel.py == f37db321`, `mixed_trellis.py == e2e645f5`
(verified by input/output hashes; scripts under
`sparkrun-glm53-exl3/overlay/patch_r22_v11/13/14/15/16/19/20.py`).

### 2.1 Per-MoE-layer launch chain (decode, inside each captured graph)

For every one of the 78 routed layers, per decode step:

1. `_pack_topk_routes_small_prefix_kernel` — **single CTA** (8 warps):
   route histogram over m*8 ids (block-padded cumsum over 256 experts),
   sentinel fill of the packed-slot arena, binary search for block expert ids.
2. `_pack_topk_routes_sort_kernel` — second tiny Triton launch: atomic
   scatter of original route indices into packed slots.
3. Fused cooperative kernel (one launch, 48 CTAs = 1 CTA/SM, 256 threads):
   input rotations (v14 compact: one gate/up row per token, FC1 maps routes)
   → grid barrier → FC1 whole-tile persistent GEMM (+ v20: ragged final wave
   striped along K through the stock split-K slice/lock/fc1_scratch finalize)
   → grid barrier → fused SiLU-mul activation + intermediate rotation epilogue
   → grid barrier → (output zero when fused-sum) → FC2 persistent GEMM
   (group4 at M32 prefill; M8 route-packed decode) writing **token-major**
   per-route rows into `buffers.fc2`.
4. `W4A16TopKSumKernel` — one CTA per (token, H128) slab: weighted top-k
   reduction of the per-route FC2 rows + down-SVH rotation + dtype-narrowed
   store (v11).

So decode pays **4 kernel launches + 3 grid-wide barriers per layer** plus the
dense/shared-expert, router, and attention launches. Route-pack work
(steps 1-2) occupies ~1-2 CTAs while 47 SMs idle; it is strictly serialized
before the fused kernel can resolve route blocks.

### 2.2 Persistent scheduler mechanics (verified in source)

- `_run_persistent_gemm` is a device-side work-state machine: whole-tile waves
  (`task = cta + wave*grid_x`) with `route_block_idx < route_blocks` bounding,
  then the lock-ordered split-K tail (`iters = ceil(k_tiles*tail/grid_x)` cells
  per CTA, `fc1_scratch` fp32 turns, final-slice bf16 store). v20 adds exactly
  one branch: when a ragged remainder exists with ≥1 completed wave, hand only
  the remainder to the tail stripe; exact fills and single-wave batches keep
  the stock schedule. All decisions are device-side (graph-replay safe).
- Wave geometry at TP4 (decode): FC1 N128×8 tiles ⇒ 8 mn-tiles/route-block,
  K-tiles 6144/128 = 48; FC2 N512×12, K-tiles 512/32 = 16. One token → 8
  distinct experts = 8 blocks = 64 FC1 tiles (1 full 48-wave + 16 ragged;
  v20 stripes the 16) + 96 FC2 tiles (2 exact waves).
- `blocks_per_sm` is **pinned to 1 for `uses_m_block_8`** in
  `_determine_blocks_per_sm` (kernel.py `485-495`): rationale is barrier-atomic
  participant count, "no extra GEMM throughput" from a bigger grid. The M8
  register entries (118-120 regs @ 256 threads) would resource-admit 2 CTAs/SM;
  M16/M32 (175/255) cannot.
- Decode is **expert-weight-stream bound** [estimate]: per step per rank the
  routed layers read ~(routes→distinct experts) × (3.5 MB FC1 + 1.8 MB FC2 at
  3.42 bpw mixed K); at 4 verification rows * 8 routes ≈ ≤32 distinct experts ≈
  up to ~170 MB/layer ⇒ tens of ms/step across 78 layers. Everything else
  (launches, barriers, pack, topk_sum, rotations) is a few-percent overlay on
  that stream. Prefill by contrast reads all 256 experts (~1.4 GB/layer/rank)
  but is dominated by attention at the 8k sample; MoE is a minor prefill cost.

## 3. Streamlining candidates (ranked; none implemented)

Ground rules kept from the repo: measured-register gate first (no guessed
register entries), narrow geometry guards, compile-time switch + cache key,
one env flip at a time, numerical-tolerance gates for any reordered reduction.

### C1 — Decode: 2 CTAs/SM for M8 plans (`blocks_per_sm=2`)

The single highest-leverage decode experiment now that v20 exists. The v20 tail
split exists because 16 CTAs could not use the whole machine; the same logic
says 48 CTAs × 4-stage pipeline may under-fill GB10's UMA bandwidth
(~48×4×(9 KB B + 2 KB A) ≈ 2 MB in flight [estimate] vs ~250-273 GB/s × ~1 µs
latency ≈ 250 KB+ needed — margin is thin, so only measurement decides).
Risk: doubles barrier participants (96 vs 48); wait-free sense-reversal barrier
cost scales with grid; FC1 64 tiles = 1 wave at 96 CTAs for the common
1-token-8-expert shape, FC2 96 tiles = exactly 1 wave — wave count halves for
cc-1 MTP3 shapes if 32-token verify batches are rare.
First gates: (a) confirmed ptxas resource numbers for the exact M8 mixed
variant; (b) run the v20 smoke shape ladder with the switch on (it already
prints per-shape timings + fc1_mn_tiles); (c) verify cooperative-barrier
residency at 96 CTAs on GB10. Keep FC2 grouping and prefill plans untouched.

### C2 — Decode: fuse the route pack into the fused kernel prologue

Steps (1)+(2) of §2.1 are two tiny launches per layer × 78 layers/step inside
the graph. Options: (a) cooperative pack in the fused prologue: grid-stride
histogram by all CTAs → grid barrier → CTA0 prefix/block-ids → grid barrier →
rotations; replaces 2 launches with 2 barriers; (b) cheaper: keep two launches
but parallelize the pack (48-CTA histogram + CTA0 prefix) to cut its serial
time. Prior art inside the same pinned b12x: the NVFP4 GLM-5.3-Flash M8 path
already split route prep into a parallel per-(token,route) pack + non-
cooperative compute (`b12x/moe/_shared/kernels/m8_route_pack.py`,
`docs/glm53_m8_route_compute_qualification.md`: +4.95% at concurrency 8,
neutral at cc-1 on the NVFP4 target). First gate: nsys trace of one captured
decode step to price the two pack launches + inter-kernel gaps vs the barrier
cost, before writing any kernel.

### C3 — Decode+prefill: fold topk_sum into the FC2 epilogue
(mixed route-packed variant of the existing `tc_decode_fused_sum`)

Today FC2 writes token-major per-route rows (`m*topk × 6144` bf16) and
`W4A16TopKSumKernel` re-reads them. The homogeneous kernel already has the
fused atomic per-token accumulation + in-kernel output pre-zero
(`tc_zero_output`); mixed explicitly rejects it
(`W4A16MixedTrellisKernel.__init__`). The store cursor knows the original
route index (`sh_route_off`) and the route weight, so a route-packed fused-sum
epilogue is expressible: multiply by w_r and atomically accumulate into
`output[token]`.
- Decode win: deletes one launch per layer (~78/step) and the fc2-buffer
  round-trip (tiny at m≤32).
- Prefill win [estimate]: at 4096 rows the buffer is ~402 MB written + read per
  layer per rank; fused atomics collapse that to ~50 MB output traffic —
  but prefill MoE is a minor share of total time (see §2.2), so treat prefill
  gain as secondary.
- Open design choices: bf16x2 atomic adds (precision gate needed) vs fp32
  accumulation buffer + a conversion the next op must absorb; SVH linearity
  only holds under `broadcast_svh` (the coupled+broadcast path already relies
  on it), so per-expert SVH must be applied per route otherwise.
First gate: isolated ptxas register/spill inspection of a dual-duty FC2 M8
epilogue; reject spilling variants. Same gate as the mapped v21 FC2
weight-reuse candidate; do not stack both in one variant.

### C4 — Prefill: FC2 decoded-weight reuse across group subtiles (mapped v21)

`_dispatch_tier_gemm` calls full `_run_tile` per M8 subtile (factor=4 with
`FC2_GROUP=4` at M32) — each restages B and re-runs the trellis LUT dequant of
the same weight tile. A two-subtile, one-B-stream/dual-accumulator variant
halves that work for those tiles. Unchanged recommendation: ptxas gate first
(current FC2 M8 entry: 118 regs; extra live accumulators risk spills).

### C5 — Prefill attention path (only if long-context prefill ever matters)

Safe's R7 `DCP_QUERY_SPLIT` (≥8192 ctx) has no R22 counterpart; indexers
diverge at ≥32k contexts (64→2 calls). Not worth action for the 8k sample.

### Non-candidates (checked, leave alone)

- M16/M32/M64 2-CTA occupancy: 2 CTAs/SM at 256 threads needs ≤128 regs/thread
  (64 K regs/SM); the measured entries (M16 158, M32 175, M64 255) all exceed
  it. Confirmed non-candidate; only M8 (118-120) has headroom.
- `_STAGES=4` pipeline depth: registers, not smem, bind; unchanged.
- Expert parallelism: shelved per v19 findings (adapter rejects expert_map;
  ~3.60/4 nodes visited/token; no dispatch/combine protocol in RoCEnante).
- Per-call Python in `run_bound_mixed_trellis` during eager prefill (binding
  cache key + ~30 validations per layer-call): ~sub-10 ms per 4096-token chunk
  [estimate] vs multi-second chunks — under 0.5%, not worth a hash-pinned
  change now.

## 4. Ordered next actions

1. Target-only decode measurement on v20 (MTP disabled, same prompts), to
   anchor every decode-side claim before more kernel work.
2. Capture the safe-image source hashes (§1.5) on the next Spark session;
   record them here and in `CURRENT_WORK.md`.
3. C1 spike: run the v20 smoke shape ladder with a throwaway `blocks_per_sm=2`
   build for M8 decode plans on one Spark; keep if the ladder shows ≥
   consistent wins, then hash-pin as v21 with the standard gates.
4. C2 prep: one nsys trace of a captured decode step naming the four
   per-layer launches and their gaps; decide pack-fusion vs parallel-pack.
5. C3/C4 share the ptxas-first gate; sequence after C1/C2 evidence, one
   candidate per version, rollback recipe each time.
