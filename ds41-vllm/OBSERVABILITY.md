# Observing the existing image without rebuilding

Use observe.py to wrap a foreground benchmark command. It saves before/after
Prometheus metrics, per-host IB/RoCE counters and host memory/pressure snapshots,
container inspection (including actual image ID/arguments), worker logs and
benchmark stdout/stderr. It does not reset any counters. Collection failures
are written into the artifacts; inspect them before drawing conclusions.
Output directories must be new to prevent overwriting a previous run.

Copy observe.py and the updated fleet.py to the Spark. An unprofiled run needs
no restart. Set CONFIG to the actual active config (including your 0.85, k0/k5,
OMP and batch settings). Do not regenerate the config for this comparison.

```bash
CONFIG=.build/cluster-c16-latest.json
python3 observe.py --config "$CONFIG" --output .build/observe-c8 -- \
  python3 benchmark.py --config "$CONFIG" --input-tokens 256 \
  --max-tokens 256 --concurrency 8 --requests 8 --output .build/bench-c8.json
```

Repeat separately at c1/c8/c16. For prefill, use concurrency1, input8192 or65536,
and short output. benchmark.py includes exact prompt token counts, per-request
TTFT, streaming decode rate and aggregate output/wall time. The latter includes
prefill and is not identical to a decode-only metric. Early EOS can shorten a
run; check completion lengths. Unique prompts avoid deliberate prefix reuse.
You can wrap the existing benchmark executable instead; all workloads must
finish before the wrapped command exits. Avoid unrelated serving traffic.

## Short CPU/CUDA trace

Copy the actual active JSON to a new filename and set only torch_profile=true.
Keep image, k, OMP, memory utilization0.85 and batch size unchanged. Restart
with that config. fleet.py adds the built-in --profiler-config for Torch,
CPU/CUDA traces, shapes on, Python stacks and memory tracing off. No image
rebuild, kernel patch or extra package is required by this integration.

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path('.build/cluster-c16-latest.json') # your actual active profile
c = json.loads(p.read_text())
c['torch_profile'] = True
Path('.build/cluster-observe.json').write_text(json.dumps(c, indent=2)+'\n')
PY
python3 fleet.py --config .build/cluster-observe.json stop
python3 fleet.py --config .build/cluster-observe.json start
python3 observe.py --config .build/cluster-observe.json --output .build/trace-c8 --profile -- \
  python3 benchmark.py --config .build/cluster-observe.json --input-tokens 256 \
  --max-tokens 64 --concurrency 8 --requests 8 --output .build/trace-bench-c8.json
```

The collector POSTs /start_profile and /stop_profile and copies the host cache's
profiles directory from each node to separate local rank directories. Existing
traces are retained and may be copied too; correlate timestamps. Traces persist
under <cache_path>/profiles on each host even if scp fails. Inspect stop-profile
and trace-copy files for errors. Profiler endpoints are exposed on the same
server interface while enabled. Restore the original config after tracing.

## Interpretation and limits

- Port data counters are normalized from standard IB units (four bytes per
  increment); deltas cover the entire collection interval, including profiling
  export overhead when enabled. They are not a pure decode bandwidth metric.
- Counters include other port traffic, NCCL and RoCEnante. Byte balance across
  rails does not prove good latency. Retry/error/discard/stall counters, when
  provided by the driver, can indicate transport trouble; raw names/units are
  preserved. Counter resets/wraps are flagged, not interpreted as rates.
- /metrics captures exposed TTFT, inter-token latency, token counts, cache and
  speculative metrics. Subtract compatible counters/histograms; do not subtract
  gauges or assume metrics missing from this build are available.
- Torch traces can show CPU CUDA launches, device kernels, copies and waits.
  CUDA graph replay may obscure detailed shapes/kernel attribution depending on
  CUPTI support. Native CPU proxy CQ polling is not a Torch operator and will
  not acquire CPU spans automatically. No live b12x proxy-stat API is exposed by
  the current adapter; running a separate process would inspect another runtime.
- Identify wait-heavy transport kernels versus compute, Engram copy/I/O gaps,
  and idle gaps. Kernel elapsed time is not all useful compute: a polling kernel
  can be waiting for a peer. Summed overlapping times are not wall-clock time.
- Do not compare profiled throughput with unprofiled baseline. Take short traces
  separately. This tool gathers evidence, not automatic bottleneck attribution.

Upstream checked 2026-09-13 20:38 UTC: heads advanced to JJ fa6ae921 and b12x
135c9715. Pins remain JJ9342b1a/b12xfd3c638 as requested; these launch hooks use
APIs verified in that existing image source. No upstream source patch added.
CPU tests pass; the collector/profiler has not been exercised on the Spark fleet.
