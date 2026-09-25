# GB10 display-reserve memory unlock — analysis

Analysis of the memory technique in the NVIDIA developer forums post
[*DeepSeek V4.1 Flash for 2x DGX Spark EXL3 3bpw, ~3M KV cache, C6, NEW: +2GB free RAM
Unlock for all GB10s*](https://forums.developer.nvidia.com/t/deepseek-v4-1-flash-for-2x-dgx-spark-exl3-3bpw-3m-kv-cache-c6-new-2gb-free-ram-unlock-for-all-gb10s/383583)
(topic 383583, posted 2026-09-18 by `emihuang` / "emi"), plus its reference implementation at
[coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark](https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark).

The model in that post (DeepSeek V4.1 Flash, EXL3 3bpw, TP2) is **not** the point here and is
explicitly out of scope for our TP4 fleet. This document covers only the memory technique.

- Reproduction procedure: [`runbook.md`](runbook.md)
- Our own hardware probe: [`probe.c`](probe.c)
- Fetch the upstream allocator (pinned + hash-verified, not vendored): `./fetch-upstream.sh`

**Status: not executed against our Sparks.** Nothing here has been run on the fleet. Everything below
is derived from reading the shipped source, the upstream repository, and the public documentation; the parts
that can only be settled on real hardware are called out as open questions.

---

## 1. Verdict

**The mechanism is real and the code does what the author says — but the "+2 GB free RAM unlock"
headline is misleading, and the useful amount is 1.75 GiB per GPU, not 2 GB.**

Summary of what holds up and what does not:

| Claim | Verdict |
| --- | --- |
| There is ~2 GB of display-reserved memory on a GB10 that Linux/CUDA cannot normally use | **Plausible and consistent with source.** The author's own docs are explicit that the reservation is firmware-mandated and set in UEFI (2 GB default, 4 GB option). |
| That memory can be made CUDA-accessible | **Real in principle, and the shipped code is coherent.** A DRM dumb buffer is mapped into the process and registered with CUDA as IO memory. |
| "Unlocking 2 GB of RAM per Spark" | **Overstated.** It is *not* added physical RAM, it is *not* ordinary `cudaMalloc` budget, and it is *not* available to arbitrary processes. The author says so himself in the repo. |
| "For all headless GB10s" | **Not demonstrated.** Verified on GB10 with driver 580.173.02 only. The repo's own validation notes mark a clean two-node install as *not yet performed*. |
| "It's like that download more RAM meme but it's for real" | **Half true.** The memory is physically real and was always there; what is new is that it is *reachable* by CUDA. Nothing is downloaded and nothing is added. |

The author's repository is unusually careful here. Its `docs/display-memory.md` states plainly: *"It does
not increase physical RAM or turn 2 GiB into ordinary CUDA memory"*, and *"Its raw read bandwidth measured
below ordinary CUDA-backed RAM in our probes."* The forum post's framing is looser than the repository's.

---

## 2. What the post actually claims

The post releases two things: a DeepSeek V4.1 Flash TP2 recipe, and the memory technique. For the memory
technique it claims:

1. On headless DGX Sparks, the ~2 GB of display-reserved memory "currently mandated by firmware" is
   allocatable by CUDA.
2. It requires restarting `nvidia_drm` with `modeset=1 fbdev=0`.
3. The amount usable is 1.75 GB, leaving ~250 MB spare; the recipe uses 1.75 GB and keeps 250 MB back.
4. It works for "the KV cache, weights, or anything CUDA can allocate", and does not have to be contiguous.

Claims 1–3 are supported by the shipped code. Claim 4 is the weakest: "anything CUDA can allocate" is not
what the code does — see §5.

---

## 3. The exact mechanism

Five stages. Everything below is read from the shipped `display_kv.c` (allocator) and
`serving/ds41/display_kv.py` (integration), plus `docs/display-memory.md` and the runtime `README.md`.

### Stage 1 — Host preparation (administrator, one-time)

The reservation is a firmware carveout, so the host work is only about making the *DRM device* usable and
keeping the framebuffer console out of the region:

```bash
sudo systemctl set-default multi-user.target           # headless, persistent
echo 'options nvidia_drm modeset=1 fbdev=0' | sudo tee /etc/modprobe.d/ds41-display-kv.conf
sudo update-initramfs -u -k all
sudo reboot
```

Required end state, verified read-only:

```bash
sudo cat /sys/module/nvidia_drm/parameters/modeset   # Y
sudo cat /sys/module/nvidia_drm/parameters/fbdev     # N
```

The UEFI display-memory reservation must stay at **2 GB**; setting it to zero removes the region the
technique depends on. The author's guarded no-reboot alternative refuses to `rmmod nvidia_drm` unless
`nvidia-smi --query-compute-apps=pid` is empty *and* `/sys/module/nvidia_drm/refcnt` is 0.

**Correction to the author's stated causality.** The post presents `fbdev=0` as the step that "accesses this
memory", and that is not what the driver source shows. `nv_drm_dumb_create()` in NVIDIA's
`open-gpu-kernel-modules` sets the allocation type to `NVKMS_KAPI_ALLOCATION_TYPE_SCANOUT` unconditionally —
there is no `fbdev`-dependent branch — so dumb buffers are served by the same scanout path whether `fbdev` is
0 or 1. The documented default is also `fbdev=1`, not 0. What `fbdev=0` really does is (a) release the
fbcon/module reference that blocks `rmmod`, and (b) avoid a competing scanout allocation owned by the fbdev
console. That allocation is mode-sized (typically a few MB to tens of MB), **not** ~2 GB, so `fbdev=0` is not
what frees the 2 GB. Treat the required state as *necessary for a clean, unloadable, headless setup*, not as the
mechanism that unlocks the memory.

The trigger for the unlock is instead `modeset=1` (the KMS dumb-buffer API must exist at all) plus the userspace
registration in stage 3. This distinction matters for reproduction: if a node is already headless with no desktop, the
memory may already be reachable with the module state unchanged.

### Stage 2 — Allocate a DRM dumb buffer inside the carveout

In `ds41_display_create()`, against a freshly reserved private VA span:

```c
p->fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);

struct drm_mode_create_dumb c = { .width = 4096, .height = display / 16384, .bpp = 32 };
ioctl(p->fd, DRM_IOCTL_MODE_CREATE_DUMB, &c);      /* display = 1792 MiB */

struct drm_mode_map_dumb m = { .handle = p->handle };
ioctl(p->fd, DRM_IOCTL_MODE_MAP_DUMB, &m);

p->base = mmap(NULL, p->total, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);   /* reserve VA */
mmap(p->base, ordinary, PROT_READ | PROT_WRITE,
     MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);                          /* ordinary prefix */
void *display_base = (char *)p->base + ordinary;
mmap(display_base, display, PROT_READ | PROT_WRITE,
     MAP_FIXED | MAP_SHARED, p->fd, m.offset);                                 /* carveout suffix */
```

The geometry is chosen so the buffer is exactly the requested size: stride is `width * bpp / 8` = 4096 × 4 =
16384 bytes/row and height is `1792 MiB / 16384` = 114688 rows, so `stride * height` is exactly
1,879,048,192 bytes. **Verified arithmetically**, and the kernel's own
`args->pitch = roundup(width * ((bpp+7)>>3), pitchAlignment)` keeps it exact at this width.

That a dumb buffer really comes from the display reservation is **corroborated by driver code**, not just asserted:
`nv_drm_dumb_create()` sets `allocParams.type = NVKMS_KAPI_ALLOCATION_TYPE_SCANOUT` and
`allocParams.noDisplayCaching = true` before calling `nvKms->allocateMemory()`, and the NVIDIA kernel logs a
GB10B-specific scanout carveout heap (`memmgrAllocScanoutCarveoutRegionResources_GB10B: EheapAlloc returns error
0x51 ... NV_ERR_NO_MEMORY`) that can be exhausted while ~118 GB of RAM is still free. That is the strongest
available evidence that scanout allocations come from a separate, finite carveout pool rather than ordinary RAM.

### Stage 3 — Register the mapping with CUDA

```c
cuMemHostRegister(p->base, ordinary, CU_MEMHOSTREGISTER_DEVICEMAP);
cuMemHostRegister(display_base, display,
                  CU_MEMHOSTREGISTER_DEVICEMAP | CU_MEMHOSTREGISTER_IOMEMORY);
cuMemHostGetDevicePointer(&display_gpu, display_base, 0);
```

`CU_MEMHOSTREGISTER_IOMEMORY` tells the driver this is device/IO memory. Per the CUDA driver API:
*"The pointer is treated as pointing to some I/O memory space, e.g. the PCI Express resource of a 3rd
party device."* It contains no page-locking language, so it does not pin the region (inference, not a literal
quotation), and registration never copies. `CU_MEMHOSTREGISTER_DEVICEMAP` *"maps the allocation into the CUDA
address space"*, which is what makes `cuMemHostGetDevicePointer()` able to hand back a device pointer.

Three documented conditions of the API matter here and the forum post omits all of them:

- **The context must be created with `CU_CTX_MAP_HOST`** for `DEVICEMAP` to have any effect. The allocator only
  requires a current context on the caller thread; the consumer must create that context with the mapping flag.
- **The host/device pointer identity is a documented device attribute, not a universal guarantee.** The driver says the
  device pointer "may or may not match the original host pointer", and only devices with a non-zero
  `CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM` let you use the host pointer on the device.
  This is why stage 4's hard check exists, and why the technique aborts rather than degrading.
- **`cuMemHostRegister` requires `CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED`**, i.e. an I/O-coherent platform. The
  CUDA for Tegra application note states it *"is supported only on platforms which are I/O coherent"* — true for GB10
  (compute capability 12.1), so the prerequisite is genuinely satisfied here rather than assumed.

**This is the documented explanation of the bandwidth penalty.** The same application note §3.2: *"if the flags for this
call specify that the memory is to be treated as memory-mapped I/O space using `cudaHostRegisterIoMemory` /
`CU_MEMHOSTREGISTER_IOMEMORY`, then the GPU L2 caching is not enabled."* So the region is **never L2-cached on the
GPU** — the author's measured slowdown is a documented consequence of the flag he uses, not an anomaly. The runtime API adds
that such memory is *"marked as non cache-coherent and contiguous"*. Any design that puts hot KV in this region pays this
cost; `probe.c` measures it rather than assuming it is small.

One honest semantic caveat: calling this "IO memory" is *defensible but approximate*. The region is on-package LPDDR5X SoC DRAM,
not the PCIe BAR the CUDA doc uses as its example; and the driver's own aarch64 path notes that video-memory allocations
"may also be kernel managed system memory, requiring `vmap()` instead of `ioremap()`", which could make the pages ordinary
struct-page-backed RAM. That is genuine uncertainty about whether the `IOMEMORY` label is literally correct here, and it is the
main reason `probe.c` measures the region rather than trusting the label.

### Stage 4 — The UVA invariant (this is why it is GB10-specific)

```c
if ((ordinary && ordinary_gpu != (CUdeviceptr)(uintptr_t)p->base) ||
    display_gpu != (CUdeviceptr)(uintptr_t)display_base) {
    snprintf(error_text, ..., "Driver did not preserve contiguous UVA: ...");
    goto fail;
}
```

The allocator **hard-fails unless the device pointer equals the host pointer**. That only holds where
host and device share an address space and the device can reach host physical addresses — i.e. GB10 unified
memory. On a discrete GPU this check fails and the technique cannot work as written. This is the single most
important structural reason the technique is GB10-specific rather than general.

### Stage 5 — Hand the pointer to the consumer

`ds41_display_pointer()` returns the span base as a `CUdeviceptr`, exposed to PyTorch through
`__cuda_array_interface__`:

```python
self.__cuda_array_interface__ = {'shape': (self.size,), 'strides': None,
    'typestr': '|i1', 'data': (self.pointer, False), 'version': 3}
tensor = torch.as_tensor(self, device='cuda:0')
if tensor.data_ptr() != self.pointer or tensor.dtype != torch.int8 or tensor.numel() != self.size:
    raise RuntimeError('CUDA array interface copied or misinterpreted external storage')
```

It then patches vLLM's native `allocate_kv_cache` and substitutes a backing function, so KV tensors
are slices of the external span instead of `torch.zeros(...)`:

```python
patched = _compile(original, [
    ('buf = torch.zeros(buf_size, dtype=torch.int8, device=device)',
     'buf = _ds41_display_backing(buf_size, dtype=torch.int8, device=device)'),
], {'_ds41_display_backing': backing})
```

The ordinary prefix exists so the process can allocate a *small* temporary profile buffer and still present
one contiguous UVA span; the **final KV uses zero ordinary RAM** (`ordinary = 0`, enforced).

---

## 4. Is it real? Evidence assessment

### Supported

- **The reservation is officially documented, and the 2 GB/4 GB claim is correct.** NVIDIA's
  [DGX Spark release notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html) state: *"Adjustable Display
  Reserved Memory: The Display Reserved Memory can now be toggled between 2GB (default) and 4GB through the system
  BIOS."* An NVIDIA employee confirms it shipped as a fix in
  [thread 370458](https://forums.developer.nvidia.com/t/nv-err-no-memory-despite-having-plenty-of-memory-available-when-using-sway/370458).
  Two caveats for our fleet: the setting does **not** appear in NVIDIA's
  [DGX Spark UEFI user guide](https://docs.nvidia.com/dgx/dgx-spark-uefi/advanced-tab.html), and a user with ASUS GX10s
  reported the option absent from their BIOS — so vendor availability is uneven and must be checked per node.
- **The reservation really is outside OS-usable RAM.** [Thread 363849](https://forums.developer.nvidia.com/t/difference-in-total-vram-available-for-different-sparks/363849)
  shows OEMs reporting different `MemTotal` (121 GiB vs 119 GiB) with the accepted answer attributing the difference to
  *"firmware carveout"*. Since the carveout is subtracted from `MemTotal`, it is outside the OS pool and therefore outside
  CUDA's allocatable pool on unified memory. The strongest direct evidence is an NVIDIA kernel log from
  [thread 370458](https://forums.developer.nvidia.com/t/nv-err-no-memory-despite-having-plenty-of-memory-available-when-using-sway/370458):
  `memmgrAllocScanoutCarveoutRegionResources_GB10B: EheapAlloc returns error 0x51 ... NV_ERR_NO_MEMORY` while only
  ~3.6 GB of 122 GB was in use — a separate, finite scanout pool that can be exhausted while RAM is free.
- **Dumb buffers really are scanout allocations.** `nv_drm_dumb_create()` sets
  `allocParams.type = NVKMS_KAPI_ALLOCATION_TYPE_SCANOUT`, and NVIDIA's DRM KMS documentation confirms the
  `DRM_IOCTL_MODE_CREATE_DUMB` / `MAP_DUMB` / `DESTROY_DUMB` mechanism as supported. Stage 2's claim is therefore
  grounded in driver code, not inferred from the author's results.
- **The allocator is complete, self-contained and matches its binary.** `display_kv.c` builds a working
  implementation; the shipped prebuilt `libds41_display_kv.so` is an aarch64 ELF that dynamically links only
  `libc.so.6` and `libcuda.so.1`, exports exactly the five symbols the source defines
  (`ds41_display_create/destroy/pointer/size/error`), and its SHA-256 matches the hash pinned in the
  repository's own `serving/overlay-manifest.json`. Source and binary are consistent.
- **The CUDA flags are used as documented.** Registering IO memory to avoid pinning/copying is the documented
  purpose of the IOMEMORY flag; this is not an abuse of the API but the intended path for device memory.
- **The technique does not fake the accounting.** The integration explicitly measures `/proc/meminfo` before and
  after and *fails* if `MemAvailable` drops by more than the ordinary prefix. That is an honest self-check that
  the 1.75 GiB did not come out of ordinary RAM.
- **The author documents limits against his own interest**, including the bandwidth penalty, the driver/firmware
  specificity, and the fact that a clean two-node install was never completed.

### Not supported / overstated

- **"2 GB unlock" is 1.75 GiB in practice, and it is not RAM.** The allocator hard-requires exactly
  1,792 MiB of display memory and at most 1 GiB ordinary. The author's own docs: *"It does not increase
  physical RAM or turn 2 GiB into ordinary CUDA memory."* The 256 MB remainder is explicitly *not* usable
  headroom. Nothing about `nvidia-smi` or `cuMemGetInfo` totals changes; if you expect the number in
  `nvidia-smi` to grow by 2 GB, you will be disappointed.
- **`c.size != display` proves nothing.** That check looks like validation but is a tautology: the ioctl
  returns the size determined by the geometry just requested, so it always matches unless the driver alters
  geometry. The real proof is the `/proc/meminfo` delta plus CUDA write/readback (what `probe.c` measures).
- **The validation probe is not shipped.** `display_kv.c` ends with `#ifdef DS41_DISPLAY_PROBE_MAIN`,
  which declares `int display_gpu_probe(CUdeviceptr, size_t);` and calls it — but that function is defined
  nowhere in the public repository (`grep` finds only the declaration and the call). The author's actual
  hardware-validation probe is private. Reproducing the validation means writing that probe ourselves; `probe.c`
  in this directory is exactly that.
- **"Anything CUDA can allocate" is too strong.** What is demonstrated is that a *registered mapping* can back
  a tensor and serve as KV cache. The documented flags only "map the allocation into the CUDA address space" and hand
  back a device pointer; **no CUDA API is documented to make `cuMemAlloc`, `cuMemCreate` or `cuMemAllocManaged`
  allocate out of a host-registered region**, and `cuMemGetInfo` reports the CUDA allocation pool, which
  host-registered memory is not part of. So you cannot `cudaMalloc` from the carveout — you take the raw device
  pointer and place your own buffers there, which is exactly what the author does. The integration patches one specific
  allocator path (`vllm.v1.worker.utils.allocate_kv_cache`), pinned by SHA-256, and refuses unpinned targets:
  *"Unreviewed native display-KV allocation target"*. In our stack it would need the equivalent hook.
- **No independent confirmation that the unlock works exists.** The only third-party reproduction is
  [issue #1](https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark/issues/1), which reproduced the
  *prerequisite state* (headless, `modeset=1 fbdev=0`) on two real Sparks across six launches, two driver versions and
  single/dual rail — then hit a 100%-reproducible vLLM/NCCL process-group-init hang at ~0% GPU utilisation, so it never
  reached serving and **never validated the extra memory**. The forum thread's only reply is "Worth a try." Searching for
  "1.75 GiB", "display reservation" or "display-backed" returns only the author's own thread. No report disputes the
  claim either. **The memory unlock rests on a single author.**
- **"For all headless GB10s" is unproven.** Verified combination is GB10 + driver **580.173.02** only; the
  repo says other driver/firmware combinations are *not qualified*. Nothing in the public repo shows a second
  independent party reproducing it, and the repository's own `docs/release-validation.md` still lists a clean
  two-node install as **not yet performed**.
- **There is a real performance cost.** Both the README and `docs/display-memory.md` state that raw read
  bandwidth is *lower than ordinary CUDA-backed memory* in the author's probes. The author's compensating claim
  (it "hardly hurts prefill/decode" for this workload) is his own measurement on his own model, not a
  general result. `probe.c` measures this penalty explicitly.
- **It is not headroom for a general serving stack.** How the region behaves when vLLM or PyTorch also wants
  ordinary memory is workload-specific, and the author warns the recipe is "very tight" at 0.92 utilization with
  nothing else running.
- **The `+2GB` figure and a 4 GB UEFI setting are unrelated.** With the reservation at 4 GB, the amount this
  technique can reach is not established by anything published.

### Open questions only real hardware can settle

1. Does `nvidia_drm` on our DGX OS accept a DRM dumb buffer of 4096 × 114688? Pixels = 469,762,048, which
   is an unusual "framebuffer" size; a driver-side cap on dumb-buffer dimensions would break stage 2.
2. Is the dumb buffer really served from the carveout, or from ordinary system memory? The `/proc/meminfo` check
   answers this indirectly; a cleaner test is whether `MemAvailable` holds while the buffer is registered.
3. What is the actual bandwidth penalty on our driver? `probe.c` reports both numbers.
4. Does a multi-GPU TP4 run behave per-rank as expected, and what does the extra 1.75 GiB per node buy in KV
   tokens for our DeepSeek V4.1 configuration?

---

## 5. Relevance to our fleet

The unlock is **per GPU, not per cluster**, so in principle it applies to our TP4 topology unchanged: four nodes ×
1.75 GiB = **7 GiB of extra KV backing across the fleet**. The parts that are specific to the author's recipe —
EXL3 quants, DSpark drafter, FP4 KV, `vllm_dcp`, six sequence slots — are irrelevant to us.

What is *not* reusable as-is is the integration: the author patches vLLM's `allocate_kv_cache` and pins the
target module by SHA-256. Our stack is a different vLLM build (`ds41-vllm`, JJ + b12x), so the same idea has to be
hooked into whatever our build uses to create KV tensors. The allocator, the host preparation and the CUDA
registration are portable; the integration is not.

Two additional portability notes we should check before investing effort:

- **Container access.** Their runtime adds `--device=/dev/dri/card0` plus `--group-add <gid>` (not a privileged
  container) and `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display`. Our launcher must expose the DRM
  node and the display capability similarly, or the allocator cannot open `/dev/dri/card0`.
- **Interaction with `--gpu-memory-utilization`.** The technique adds a KV backing that is outside the allocator's
  ordinary budget, which is exactly why the author's variant can run at 0.92 with a 1.75 GiB display-backed pool and
  zero ordinary KV. Whether our `--gpu-memory-utilization` accounting tolerates an external KV backing without
  tripping its own startup admission checks is the main integration risk and needs a test.

---

## 6. Risks and costs

| Risk | Assessment |
| --- | --- |
| Data loss / crash | Low for the technique itself, but the author warns the watchdog "cannot guarantee recovery from a hard unified-memory lockup" on a unified-memory platform. |
| Bricking via `rmmod nvidia_drm` | Real if unloaded while a desktop or CUDA job holds it. The author's guard (empty compute-apps list and `refcnt` 0) is mandatory, not optional. |
| Losing the desktop | Expected: a headless boot target disables the desktop. Requires SSH access that survives the change, and a documented rollback. |
| Performance regression | Real and acknowledged; measure with `probe.c` before committing to a design that depends on the extra memory. |
| Firmware confusion | Setting the UEFI reservation to 0 destroys the region this technique needs. Keep it at 2 GB. |
| Licensing | The allocator and binary are **AGPL-3.0-only**. If we port it, the license obligations travel with it. This repository keeps them unvendored and separately licensed on purpose. |
| Effort vs. payoff | 1.75 GiB per node for a driver-state change, a host prep step and a serving-stack patch. On TP4 that is 7 GiB of KV backing across the fleet — worth it only if long-context KV capacity is our actual constraint. |

---

## 7. Reproducing it

See [`runbook.md`](runbook.md) for a staged procedure that starts with read-only observation, then a
standalone probe on **one** node, and only then the serving-stack port. Per instruction, no step here has been
executed against our Sparks, and nothing in this directory SSHs anywhere.
