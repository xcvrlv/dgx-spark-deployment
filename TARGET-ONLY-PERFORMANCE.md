# Target-only performance log

Dedicated performance log for **target-only** runs of the GLM-5.3 EXL3 target
model on TP4 (four DGX Spark GB10/SM121 nodes): decode measured without MTP
speculative decoding, so draft/verify overhead is excluded from the decode
rate. Prefill is recorded alongside since it is largely MTP-independent.
Companion to `CURRENT_WORK.md` and `EXL3-SCHEDULING-NOTES.md`; this log holds
measurements only.

Methodology: uncached prompts for prefill (TTFT and prompt tok/s at N=1);
aggregate decode tok/s at context 0, concurrency 1/2/4/8. Each entry records
the recipe, container, model, and MTP state actually run. Compare medians,
not the first request after startup.

## Entry 1 — safe recipe (recorded 2026-09-08)

- Recipe: `sparkrun-glm53-exl3/recipes/glm53-exl3-4x-safe.yaml`
- Container: `spark-vllm-glm52-exl3:sparkring-switch-prefill-v2`
  (SparkRing R7 lineage, R7 vLLM + one hash-gated indexer patch)
- Model: `davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw`, TP4
- MTP: off (target-only run)
- Numbers supplied by the operator; tables verbatim below. This entry is the
  comparison anchor for the pending v20 target-only run (see
  `CURRENT_WORK.md` next steps).

Prefill tok/s:

| ctx  |  tokens | TTFT s | tok/s | N |
|------|---------|--------|-------|---|
| 8k   |   8,198 |  11.07 |   741 | 1 |
| 64k  |  64,492 |  85.15 |   757 | 1 |
| 128k | 128,852 | 181.20 |   711 | 1 |

Aggregate decode tok/s:

| ctx \ conc |    1 |    2 |    4 |    8 |
|------------|------|------|------|------|
| 0          | 11.6 | 22.1 | 32.2 | 56.6 |

Notes:

- Prefill holds >=700 tok/s across 8k/64k/128k (741/757/711). TTFT tracks
  tokens/tok/s exactly (8198/741 = 11.07 s), so tok/s is the TTFT-derived
  rate and TTFT carries no additional information at N=1.
- Decode scales sublinearly: 4.9x from concurrency 1 to 8.
- Historical context: this is the line v13..v20 were compared against
  ("v20 measured similar prefill and decode" to safe, per
  `CURRENT_WORK.md`). The safe recipe's serving command ships native MTP3;
  this entry is recorded as the MTP-off run state for target-only tracking -
  relabel before comparing if this run actually used MTP3.

## Entry 2 — v20 target-only (recorded 2026-09-08)

- Recipe: `sparkrun-glm53-exl3/recipes/glm53-exl3-v20-4x.yaml`
  (`VLLM_GB10_EXL3_FC1_TAILSPLIT=1`, v19 `FC2_GROUP=4`)
- Container: `spark-vllm-glm53-exl3:r22-dflash2-sm121-v20`
- Model: `davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw`, TP4
- MTP: off (target-only run)
- Decode samples: 30-second runs per cell. Numbers supplied by the operator;
  tables verbatim below.

Prefill tok/s:

| ctx  |  tokens | TTFT s | tok/s | N |
|------|---------|--------|-------|---|
| 8k   |   8,197 |  12.71 |   645 | 1 |
| 64k  |  64,477 |  99.95 |   645 | 1 |
| 128k | 128,813 | 204.96 |   628 | 1 |

Aggregate decode tok/s:

| ctx \ conc |    1 |    2 |    4 |    8 |
|------------|------|------|------|------|
| 0          | 11.1 | 20.5 | 38.5 | 59.5 |

Comparison vs entry 1 (safe, matched target-only):

- Prefill: v20 trails safe by ~96 tok/s at 8k (645 vs 741), ~112 at 64k
  (645 vs 757) and ~83 at 128k (628 vs 711). The long-standing prefill gap
  persists in the matched comparison; v20's tail split is decode-scoped, so
  prefill was expected unchanged from v19 (~650, earlier user observation).
  The safe-image prefill advantage remains unattributed.
- Decode: v20 wins at conc 4 (+19.6%) and 8 (+5.1%), trails at conc 1
  (-4.3%) and 2 (-7.2%). Single 30-second sample per cell - treat the small
  conc-1/2 deltas as within-variation candidates until repeated.
- Attribution caveat: safe runs the R7 b12x kernel lineage, so these deltas
  are v20-line-vs-safe, not TAILSPLIT-vs-stock. The clean isolation is a
  target-only A/B of `VLLM_GB10_EXL3_FC1_TAILSPLIT` 0 vs 1 on v20 (one env
  flip, restart all workers).
- The earlier [user-observed] MTP3-inclusive serving comparison ("similar
  prefill and decode") is not reproduced here at 8k prefill; the keep-v20
  decision rests on the serving comparison plus the production-concurrency
  decode wins (conc 4/8).

## Entry 3 — v20 target-only, TAILSPLIT=0 (300-second decode, recorded 2026-09-08)

- Recipe: `sparkrun-glm53-exl3/recipes/glm53-exl3-v20-4x.yaml` with
  `VLLM_GB10_EXL3_FC1_TAILSPLIT=0` (the stock v19 schedule - baseline arm of
  the clean tail-split A/B)
- Container: `spark-vllm-glm53-exl3:r22-dflash2-sm121-v20`
- Model: `davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw`, TP4
- MTP: off (target-only run)
- Decode samples: 300-second runs per cell. Duration differs from entry 2's
  30-second samples, so this row is NOT directly comparable against entry 2.
  Numbers supplied by the operator; table verbatim below.

Aggregate decode tok/s:

| ctx \ conc |    1 |    2 |    4 |    8 |
|------------|------|------|------|------|
| 0          | 10.9 | 20.0 | 38.1 | 57.8 |

Notes:

- `TAILSPLIT=0` arm of the clean isolation A/B. The 300-second safe rerun is
  recorded below (entry 5); the `TAILSPLIT=1` arm was supplied but is
  byte-identical to this entry (entry 4, UNVERIFIED) - the matched
  comparison stays unresolved until that arm is confirmed.
- Do not compare against entry 2 (30 s): any delta would be confounded by
  warmup/steady-state differences rather than the schedule.

## Entry 5 — safe target-only (300-second decode, recorded 2026-09-08)

- Recipe: `sparkrun-glm53-exl3/recipes/glm53-exl3-4x-safe.yaml`
  (target-only) - the matched safe arm of the 300-second A/B
- Container: `spark-vllm-glm52-exl3:sparkring-switch-prefill-v2`
  (SparkRing R7 lineage)
- Model: `davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw`, TP4
- MTP: off (target-only run); 300-second decode samples per cell
- Numbers supplied by the operator; table verbatim below. Prefill not
  supplied for this run.

Aggregate decode tok/s:

| ctx \ conc |    1 |    2 |    4 |    8 |
|------------|------|------|------|------|
| 0          | 12.9 | 21.8 | 35.6 | 54.1 |

Matched 300-second comparison (entry 5 vs entry 3):

- Safe vs v20 `TAILSPLIT=0` (v20-vs-safe form, matching entry 2's signs):
  v20 trails at conc 1 (-15.5%) and 2 (-9.0%); wins at conc 4 (+7.0%) and
  8 (+6.8%). Same shape as the 30-second comparison (entry 1 vs 2: v20 wins
  conc 4/8, trails at 1/2) - the pattern is duration-consistent, so the
  v20-vs-safe decode split (v20 wins at production concurrency, trails at
  1/2) holds in the matched 300 s arms.
- The safe 30-second and 300-second runs are different samples; the 300 s
  magnitudes (12.9/21.8 vs 11.6/22.1 at conc 1/2) differ from the 30 s
  sample, so do not mix durations in one attribution.
- Vs entry 4 (`TAILSPLIT=1`, UNVERIFIED, byte-identical to entry 3): the
  same deltas apply only if entry 4's table is genuine; the tail-split
  isolation stays unresolved until the engagement check or a valid rerun
  confirms the arm.

## Entry 4 — v20 target-only, TAILSPLIT=1 (300-second decode, supplied 2026-09-08 — UNVERIFIED)

- Recipe: `sparkrun-glm53-exl3/recipes/glm53-exl3-v20-4x.yaml` as configured
  (`VLLM_GB10_EXL3_FC1_TAILSPLIT=1`); all-worker restart expected for the env
  flip to recompile
- Container: `spark-vllm-glm53-exl3:r22-dflash2-sm121-v20`
- Model: `davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw`, TP4
- MTP: off (target-only run); 300-second decode samples per cell
- **UNVERIFIED: the supplied table is byte-identical to entry 3
  (`TAILSPLIT=0`, 10.9/20.0/38.1/57.8).** Either the `=0` table was re-pasted
  by mistake, or the env flip did not take effect (both arms ran the same
  schedule), or the schedule change is a genuine null at these shapes.
  Identical values to three significant figures at all four concurrencies is
  not a plausible run-to-run outcome for a schedule that alters the FC1
  ragged-wave finalize path (at ctx-0 conc-1 the 16-tile remainder engages
  the split), so treat this arm as unresolved until one of the following is
  confirmed.

Aggregate decode tok/s:

| ctx \ conc |    1 |    2 |    4 |    8 |
|------------|------|------|------|------|
| 0          | 10.9 | 20.0 | 38.1 | 57.8 |

Verification before treating this as a matched A/B:

- Engagement check **passed** (2026-09-08: with the env set to `2`, EngineCore
  raises `VLLM_GB10_EXL3_FC1_TAILSPLIT must be 0 or 1` at GEMM compile). The
  env IS read. This eliminates a broken env mechanism; the remaining
  explanations for entry 4's byte-identical table are (a) the run's env was
  `0` (the flip was not applied to the serving environment despite intent -
  e.g. the recipe edited after the run, or the env not propagated to every
  worker) or (b) a re-paste.
- Next: rerun `TAILSPLIT=1` with the env confirmed (set `1`, restart all
  workers, then rerun the intended arm) and compare against entry 3. If the
  rerun repeats entry 3's numbers, record as a null result: the tail split
  shows no measurable target-only decode effect at ctx-0 decode shapes, and
  the v20-vs-safe decode deltas (entry 2) come from elsewhere in the
  v11..v20 overlays. If the rerun differs, entry 4's table was a re-paste -
  replace it with the real `TAILSPLIT=1` table.
