# GB10 display-reserve KV — report

Consolidated report on the display-reserve memory technique published on the NVIDIA
developer forums, and what it is worth to the ds41-vllm R38 TP4 serving target.

**Status: nothing has been run on a Spark.** Everything below comes from reading the
published source, the upstream repository, NVIDIA's own documentation and the CUDA
driver sources, plus tests that run offline. No fleet capacity number has been measured.
Attribution and the reasoning behind every claim are in §3 and §8.

| Sub-question | Answer |
| --- | --- |
| Is the technique real? | **Yes.** Every ingredient maps onto a documented platform feature, and the shipped code does what the author says. |
| Is the headline claim right? | **No, it is overstated in three ways.** §3. |
| Is it useful to us? | **Probably, but for less than the headline.** 1.75 GiB/GPU, ×4 nodes = 7 GiB of KV backing. §5. |
| Can we calculate the per-node gain? | **Yes**, from one startup log. §6. |
| Is it verified on our hardware? | **No.** Every hardware check is still open. §8. |

---

## 1. The claim

The forum post is [topic 383583](https://forums.developer.nvidia.com/t/deepseek-v4-1-flash-for-2x-dgx-spark-exl3-3bpw-3m-kv-cache-c6-new-2gb-free-ram-unlock/383583),
"*DeepSeek V4.1 Flash for 2x DGX Spark EXL3 3bpw, ~3M KV cache, C6, NEW: +2GB free RAM
Unlock for all GB10s*", posted 2026-09-18 by `emihuang`. It has three posts, one reply
("*Worth a try*"), and its reference implementation is
[coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark](https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark).

Two claims matter here, and the model recipe does not (we run TP4, that recipe is TP2):
a firmware display reservation can be reached by CUDA, and this is claimed for all headless GB10s.

## 2. What the technique is

A GB10 reserves ~2 GB of memory for the display, set in UEFI (2 GB default, 4 GB option).
On a headless Spark nothing displays, so the author reaches that region with CUDA in five stages:

1. **Headless host**: `systemctl set-default multi-user.target`, and
   `options nvidia_drm modeset=1 fbdev=0`, then `update-initramfs -u -k all` and reboot.
2. **DRM dumb buffer**: `/dev/dri/card0` + `DRM_IOCTL_MODE_CREATE_DUMB` at 4096 × 114688 × 32bpp,
   which is exactly 1,879,048,192 bytes = 1.75 GiB, mapped with `MAP_FIXED` into a reserved VA span.
3. **Register with CUDA**: `cuMemHostRegister(..., CU_MEMHOSTREGISTER_DEVICEMAP | CU_MEMHOSTREGISTER_IOMEMORY)`.
4. **UVA invariant**: the allocator *hard-fails* unless `cuMemHostGetDevicePointer` returns the host
   pointer — which is why this is GB10-specific and not general.
5. **Hand the pointer to PyTorch** via `__cuda_array_interface__`, and back the KV tensors with it.

The reason it works is that the region is firmware-reserved, so Linux never counts it in `MemTotal`
and it is invisible to both the OS and `cuMemGetInfo` — but the display engine can still allocate in it
and CUDA can register that mapping.

## 3. What is verified, and what is overstated

| Claim | Verdict |
| --- | --- |
| The display reservation exists, 2 GB default / 4 GB in the BIOS | **Verified.** NVIDIA's [DGX Spark release notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html) document *"Adjustable Display Reserved Memory"*, and an NVIDIA employee confirms it shipped. Caveat: it is absent from NVIDIA's own UEFI guide, and an ASUS GX10 user reported it missing from their BIOS. |
| The reservation is outside OS-usable and CUDA-allocatable memory | **Verified.** [Thread 363849](https://forums.developer.nvidia.com/t/difference-in-total-vram-available-for-different-sparks/363849) shows OEMs at 121 vs 119 GiB `MemTotal`, attributed to firmware carveout. Strongest evidence: a GB10B `memmgrAllocScanoutCarveoutRegionResources_GB10B: EheapAlloc returns error 0x51` log while only ~3.6 GB of 122 GB was in use. |
| A DRM dumb buffer comes from that reservation | **Verified in code.** `nv_drm_dumb_create()` sets `NVKMS_KAPI_ALLOCATION_TYPE_SCANOUT` unconditionally, and NVIDIA's DRM KMS doc lists the dumb-buffer mechanism. |
| The CUDA flags are used as documented | **Verified.** `CU_MEMHOSTREGISTER_IOMEMORY` *"is treated as pointing to some I/O memory space"*; `DEVICEMAP` *"maps the allocation into the CUDA address space"*. |
| The shipped `.so` matches its source | **Verified.** aarch64 ELF linking only `libc` and `libcuda`, exporting exactly the five functions the source defines, and its SHA-256 matches the repo's own pinned overlay manifest. |
| "**+2 GB free RAM unlock for all GB10s**" | **Overstated, three ways.** It is 1.75 GiB (−14%); it is *not* RAM and *not* `cudaMalloc` budget and *not* available to every process; and no independent confirmation it works exists. §4. |
| `fbdev=0` is what accesses the memory | **Unsupported.** `nv_drm_dumb_create()` has no `fbdev` branch, and the documented default is `fbdev=1`, not 0. Its real roles are module unloadability and avoiding a competing console scanout allocation. |
| "Anything CUDA can allocate" | **Too strong.** No CUDA API is documented to make `cuMemAlloc` or `cuMemCreate` allocate out of a host-registered region; you take the raw device pointer and place buffers there. |
| `c.size != display` validates the allocation | **Tautological.** The ioctl returns the size implied by the geometry just requested, so it cannot fail. |

## 4. The three things that are overstated

1. **It is 1.75 GiB, not 2 GB, and it is not RAM.** The author's own docs say it
   *"does not increase physical RAM or turn 2 GiB into ordinary CUDA memory"* and that the nominal
   remaining 256 MiB *"is not extra host-RAM safety margin"*. `nvidia-smi` and `cuMemGetInfo` totals
   **do not change** — if the expectation is that the number grows by 2 GB, it will not.
2. **There is no independent confirmation.** The only third-party reproduction is
   [issue #1](https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark/issues/1), which reproduced
   the *prerequisite state* on two Sparks and then hit a 100%-reproducible vLLM/NCCL hang, so it never
   reached serving and never validated the extra memory. The single forum reply is "Worth a try", no report
   disputes the claim, and searching for the technique returns only the author's own thread.
   **The unlock rests on one author.**
3. **It costs performance.** `CU_MEMHOSTREGISTER_IOMEMORY` *"means GPU L2 caching is not enabled"*
   ([CUDA for Tegra §3.2](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/)) — a documented
   consequence of the flag, not an anomaly. The author agrees: *"lower raw read bandwidth than ordinary
   CUDA-backed memory in our probes"*. His mitigation is workload-specific to his model.

## 5. Where it could be used, and what it would be worth

The unlock is **per GPU**, so it applies to our TP4 target unchanged: four nodes × 1.75 GiB =
**7 GiB of extra KV backing across the fleet**, obtained **without touching `gpu_memory_utilization`
(0.85)**, so the ordinary margin the launcher deliberately keeps is preserved. Our own notes already frame
capacity this way — `UPSTREAM-R38.md`: *"Compare reported KV token capacity at the same utilization to
measure the fleet gain"*.

What ports, and what does not:

| Piece | Portable? |
| --- | --- |
| The allocator, unmodified | **Yes** — it needs only `/dev/dri/cardN` and libcuda |
| Host preparation (headless + module state) | **Yes**, per node |
| The `cuMemHostRegister(IOMEMORY)` idea | **Yes** — it is the documented API for device memory |
| The vLLM integration | **No** — it patches an exact module by SHA-256; our build is a different tree |
| KV budget and accounting changes | **No** — recipe-specific (0.92 utilization, six slots, FP4 KV) |

So this is worth doing if long-context KV capacity is our binding constraint, which our notes suggest it
is. It is **not** worth doing as a general "more free RAM" change, because that is not what it is.

## 6. The per-node number, and the 100% trap

Per node, with every term at its real source at our pin:

```
usable = ceil(total_memory × gpu_memory_utilization)      v1/worker/utils.py:505
         − non_kv_cache − late_persistent − cudagraph   v1/worker/gpu_worker.py:657
         [+ display_credit]                             additive, outside total_memory
```

`kv-budget.py` computes the whole chain from one `rank-N.log`, or from explicit numbers.

**On the 100% question.** The credit is *additive* and the display reserve is not part of `total_memory`,
so the credited claim against `total_memory` is `utilization + credit/total` — 1.5 GiB on a 121.66 GiB device is
85.00% → 86.23%, and the ordinary residual stays at 18.25 GiB. The technique needs **no** change to
`gpu_memory_utilization`. **`--kv-cache-memory-bytes` is the wrong tool**: per `entrypoints/llm.py:121` it
*"ignores gpu_memory_utilization"*, so setting it to the reported maximum claims the whole budget with no
utilization guard — exactly the 100% case to avoid.

**On how much capacity 1.75 GiB actually buys.** Reported capacity is `int(concurrency) × max_model_len`, so it is
quantised to whole max-length requests, while the credit is a fixed byte amount:

| Ordinary KV budget | one max-length request | credited | full requests gained |
| --- | --- | --- | --- |
| 100.0 GiB | 14.33 GiB | 101.5 GiB | **+1** (6.98 → 7.08) |
| 51.0 GiB | 14.40 GiB | 52.5 GiB | **0** (3.54 → 3.64) |

Both rows are correct. The same 1.5 GiB credit buys one whole 393K request in the first case and none in the
second, purely from where the floor falls. So the token gain is real but **not fixed**, and the honest measurement is
the unquantised logged budget (`Available KV cache memory`), with a flat token reading reported as the floor effect
rather than a failure.

## 7. Limitations

1. **1.75 GiB, not 2 GB**, with 256 MiB of the reservation deliberately held back.
2. **Not ordinary RAM, not `cudaMalloc` budget, not visible in `nvidia-smi`.** It is reachable only through a
   registered device pointer.
3. **Quantised to whole max-length requests**, so the token gain depends on the floor boundary (§6).
4. **Documented performance penalty**: no GPU L2 caching for this region; measured cost is still open.
5. **Driver and firmware pinned.** Measured on 580.173.02 only; the repo qualifies no other combination, so our
   driver must be checked first.
6. **Requires headless and a module-state change**, with a desktop, monitor or remote-desktop session excluded, and a
   reboot or a guarded `rmmod`. Losing the desktop is expected and needs a rollback path.
7. **Requires application-level integration.** The author is explicit: *"setting the module flags alone is not enough"* —
   allocator integration, ownership, lifetime and memory accounting are all needed.
8. **Must fail closed.** A silent fallback to `torch.zeros` would allocate the credited block count out of ordinary RAM
   at 0.85 utilization and OOM, which is worse than failing at startup.
9. **Not an allocation target**, so it cannot enlarge the `cudaMalloc` pool for anything that expects one.
10. **The unused region must stay at 2 GB in UEFI** — setting it to zero removes what the technique depends on.
11. **The 4 GB UEFI option is unestablished**: nothing published says how much of it this technique reaches.
12. **A unified-memory platform has no guaranteed recovery** from a hard lockup; the author's watchdog is an emergency
    guard, not a guarantee.
13. **`fbdev=0` may not exist as a parameter.** `fbdev` is conditional on `NV_DRM_FBDEV_AVAILABLE`, and there are real
    reports of `nvidia_drm: unknown parameter 'fbdev'`, so the state must be checked rather than assumed.
14. **Sysfs parameter permissions are not durable.** The third-party reporter found `update-initramfs -u -k all` alone did not
    keep them across reboots; a udev rule was needed.
15. **The author's validation probe is not shipped**, so reproducing the evidence means writing that probe.
16. **Both pins have moved**, and one of our two patch anchors changed at the new head, so a pin bump needs a `SOURCE_SHA`
    rebase. Neither pin should be advanced on the strength of the technique.

## 8. What is unclear

These are only settled on real hardware, and each one can invalidate part of the plan:

1. Does our `nvidia_drm` accept **one** 4096 × 114688 dumb buffer? 469,762,048 pixels is an unusual framebuffer
   size, and a driver cap would break stage 2. Fallback: split the span into several buffers mapped with `MAP_FIXED`.
2. Does `reserve_mm_ipc_gpu_memory` draw on the credited bytes? If it does, the credit must be applied after it.
3. Does B12X's attention accept a host-registered device pointer for KV, or does it assume `torch.zeros`-style memory?
4. What is the measured read penalty at our KV geometry under B12X?
5. Is `IOMEMORY` literally correct here? The region is on-package SoC DRAM, not the PCIe BAR the CUDA doc
   uses as its example, and the driver's own aarch64 branch notes video-memory allocations may be kernel-managed system
   memory — which would make the label approximate rather than literal.
6. Is the carveout a genuinely separate, finite pool on our driver, as the GB10B scanout heap path suggests?
7. Is the UEFI setting actually present on our specific OEM units?
8. How does an external KV backing interact with our `--gpu-memory-utilization` admission checks, which cannot see it?
9. What is the measured per-node number? `total_memory`, `non_kv_cache`, `late_persistent` and `cudagraph_estimate`
   are all still unmeasured, and they set the whole result.

## 9. State of the work

| Artifact | What it is |
| --- | --- |
| [`DISPLAY-KV-INTEGRATION.md`](DISPLAY-KV-INTEGRATION.md) | How to integrate it into this target: hook points, guardrails, test plan, pin policy |
| [`kv-budget.py`](kv-budget.py) | Per-node and fleet budget calculator, from a log or explicit numbers |
| [`patches/display_kv.py`](patches/display_kv.py), [`patches/display_kv_credit.py`](patches/display_kv_credit.py) | The two hash-guarded patches, with independent rollback switches |
| [`patches/ds41_display_kv.py`](patches/ds41_display_kv.py) | Runtime helper: span ownership, fail-closed, one pool, credit bound |
| [`tests/test_display_kv.py`](tests/test_display_kv.py) | Patch tests and the headroom test; 8 pass, 5 hardware-gated skips |
| [`display-check.py`](display-check.py) | Container-side fail-closed preflight check |
| `fleet.py`, `Dockerfile`, `cluster-r38-c8.json`, `configure-r38.py` | Device/group exposure, capability, credit, label, preflight; credit off by default |
| [`../gb10-display-ram/`](../gb10-display-ram/README.md) | Technique analysis, and the operator runbook in [`runbook.md`](../gb10-display-ram/runbook.md) |

Verified offline on 2026-09-19: both patch anchors present at the pin with hashes that reproduce two already-recorded
pins; re-application, `--check`, drift refusal and independent rollback all pass and the patched files compile; the credit
is inert to the byte while disabled and bounded by the allocator's span; the disabled launch plan is byte-identical to today's;
`configure-r38.py` carries `display_kv`; and 8 tests pass with 4 hardware tests skipping.

**Not verified: the capacity gain itself, and every item in §8.** No image was built, no server was started and no Spark was
contacted. A capacity claim before §8 is settled would not be attributable to the technique.

## 10. Recommendation

**Worth doing, in stages, in this order:**

1. **Settle §8 item 1** on one idle node with a standalone probe — this is the cheapest way to find out whether the
   technique even applies to our driver, and it doubles as the container preflight check.
2. **Measure the per-node number** with `kv-budget.py` on one node, and confirm the credited budget matches the prediction.
3. **Only then** port the integration into our stack, measuring one node before the fleet. The main integration risk is item 8:
   an external KV backing is invisible to our utilization-based admission.
4. **Do not** raise `gpu_memory_utilization` and **do not** switch to `--kv-cache-memory-bytes`. The credit is additive
   and is the only change needed to obtain the extra capacity safely.

If KV capacity is not actually our constraint, the honest answer is that this is not worth the driver-state change and the
serving-stack patch, and 1.75 GiB per node is not a reason on its own.
