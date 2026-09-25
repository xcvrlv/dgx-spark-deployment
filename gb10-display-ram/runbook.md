# GB10 display-reserve unlock — reproduction runbook

Operator procedure for testing the display-reserve memory technique from
[NVIDIA forum topic 383583](https://forums.developer.nvidia.com/t/deepseek-v4-1-flash-for-2x-dgx-spark-exl3-3bpw-3m-kv-cache-c6-new-2gb-free-ram-unlock/383583)
on our DGX Spark fleet. Analysis and verdict: [`README.md`](README.md).

**Nothing in this runbook has been run on our Sparks.** These are steps for an
operator to execute deliberately, one node at a time, starting with a node that is
not serving. Do not run stage 1 on a Spark that has work you care about.

Design of the sequence: stage 0 changes nothing and costs nothing; stage 1 is the only
step that needs a reboot; stages 2–4 are a self-contained measurement that does not
touch our serving stack. Only if all of those pass is a port into our vLLM stack worth
starting (stage 5).

Throughout, "the node" means one DGX Spark. The unlock is per GPU, so a single node is
a sufficient test bed and a valid result generalises to the other three.

---

## Stage 0 — read-only reconnaissance

No changes. Establishes the baseline and decides whether the technique can even apply.

```bash
# Everything the proposer's recipe requires should be recorded before changing anything.
uname -m                                            # expect aarch64
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
cat /etc/os-release | head -3

# 0a. Does the driver we run match the only qualified combination (580.173.02)?
#     The upstream repo qualifies nothing else.

# 0b. Does the firmware reservation exist at all? Compare MemTotal with 128 GiB.
#     Whatever is missing is firmware reservation, of which the display carveout is a part.
grep -E 'MemTotal|MemAvailable' /proc/meminfo
echo "128 GiB in kB = $((128 * 1024 * 1024))"

# 0c. Current display / module state. These are the values stage 1 must end at Y / N.
sudo cat /sys/module/nvidia_drm/parameters/modeset      # want Y
sudo cat /sys/module/nvidia_drm/parameters/fbdev        # want N
systemctl get-default                                   # note it; rollback needs it
systemctl is-active display-manager                      # want inactive (headless)

# 0d. Which DRM node is the NVIDIA one (vendor 0x10de)?
for c in /sys/class/drm/card[0-9]*; do
    [ -r "$c/device/vendor" ] && printf '%s vendor=%s\n' "${c##*/}" "$(cat "$c/device/vendor")"
done

# 0e. Can an ordinary user open it? Note owner/group and our groups.
ls -l /dev/dri/card0
id

# 0f. Is the node otherwise idle?
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Decision points:

- **0b:** if `MemTotal` is already ~128 GiB, there is no firmware reservation on this box
  and the technique has nothing to reclaim. Stop and re-check the UEFI setting.
- **0a:** if the driver is not 580.173.02, treat every result as unqualified and note the
  version alongside the measurement.
- **0c:** if `modeset` is already `Y` and `fbdev` already `N`, **stage 1 is not needed** —
  record the values and go to stage 2. On a properly headless Spark this is likely, since
  the repository says the required state may already be present.

---

## Stage 1 — host preparation (reboot required)

Only if stage 0c showed the state is not already correct. This is the step the author calls
the "one-time administrator setup".

```bash
# Save the rollback information first. Losing it makes stage 8 guesswork.
{
    echo "default target: $(systemctl get-default)"
    echo "modeset: $(sudo cat /sys/module/nvidia_drm/parameters/modeset)"
    echo "fbdev:   $(sudo cat /sys/module/nvidia_drm/parameters/fbdev)"
    ls /etc/modprobe.d/
} | tee ~/display-reserve-prechange.txt
```

Then make the changes:

```bash
# Headless boot. This disables the desktop, so confirm SSH access from elsewhere first.
sudo systemctl set-default multi-user.target

# Dedicated file, deliberately not editing any existing NVIDIA config.
sudo tee /etc/modprobe.d/ds41-display-kv.conf >/dev/null <<'EOF'
options nvidia_drm modeset=1 fbdev=0
EOF

# Inspect for conflicting nvidia_drm options before rebuilding.
grep -rn nvidia_drm /etc/modprobe.d/ || true

sudo update-initramfs -u -k all
sudo reboot
```

Two caveats found by an independent third party
([issue #1](https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark/issues/1)):

- `update-initramfs -u -k all` alone did **not** preserve the sysfs parameter permissions across
  reboots — `/sys/module/nvidia_drm/parameters/*` came back `644` each boot. If the probe runs as an
  ordinary user, read the parameters with `sudo` or add a udev rule; do not treat a permission error as
  evidence about the technique.
- `fbdev` only exists when the driver is built with framebuffer-device support
  (`NV_DRM_FBDEV_AVAILABLE`). There are real reports of `nvidia_drm: unknown parameter 'fbdev'` on some
  driver/GPU combinations, so check `/sys/module/nvidia_drm/parameters/` for what actually exists rather than
  assuming the parameter is there.

Note for expectation-setting: the author presents `fbdev=0` as the step that accesses the memory, but the driver
source does not support that — dumb buffers are scanout allocations regardless of `fbdev`, and the documented default
is `fbdev=1`. What `fbdev=0` buys is a clean, unloadable, headless setup (see §3 of the analysis). So a node whose
module state is unchanged but which is already headless and desktop-free may still be a valid test bed.

After reconnecting, confirm the required state:

```bash
sudo cat /sys/module/nvidia_drm/parameters/modeset   # Y
sudo cat /sys/module/nvidia_drm/parameters/fbdev     # N
systemctl is-active display-manager                   # inactive
```

Do **not** blacklist `nvidia_drm` and do **not** use `nomodeset`: the allocator needs
modesetting enabled and only the framebuffer console disabled. Do **not** set the UEFI
display reservation to zero — that removes the region.

Guarded no-reboot alternative, only while the node is fully idle. If either check fails,
stop and investigate rather than forcing it:

```bash
(
    set -eu
    test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
    test "$(sudo cat /sys/module/nvidia_drm/refcnt)" = 0
    sudo rmmod nvidia_drm
    sudo modprobe nvidia_drm modeset=1 fbdev=0
)
```

Never unload `nvidia` or `nvidia_uvm`.

---

## Stage 2 — build the probe

```bash
# Build dependencies on the node (not inside our serving container).
sudo apt-get update
sudo apt-get install -y build-essential libdrm-dev

# Fetch the pinned upstream allocator + binary, verified by SHA-256.
cd gb10-display-ram
./fetch-upstream.sh

# Read the AGPL source before running anything that links it.
less upstream/display_kv.c
less upstream/display_kv.py     # the reference PyTorch integration

# Build. If -lcuda fails because only libcuda.so.1 exists:
make check
# make CUDA_LIB='-l:libcuda.so.1' check
```

`make check` should show the allocator exporting `ds41_display_create`, `ds41_display_destroy`,
`ds41_display_pointer`, `ds41_display_size` and `ds41_display_error`, and the probe linked against
`libcuda.so.1`.

If opening `/dev/dri/card0` fails with permission denied, add the user to the group shown by
`ls -l /dev/dri/card0` and reconnect. Do not run the probe as root as a shortcut: a root-only
result would not tell us whether our serving user can use the technique.

---

## Stage 3 — run the probe (positive case)

Stop every model server and CUDA job on the node first. The measurement is only meaningful
on an idle node.

```bash
# Display-only variant (matches the technique's default KV path: zero ordinary KV).
./gb10_display_probe --ordinary-mib 0

# Or with a contiguous ordinary prefix, as the author's own probe main uses.
./gb10_display_probe --ordinary-mib 1024
```

Read the probe output as follows.

| Line | What it tells us | Expected if the technique is real |
| --- | --- | --- |
| `nvidia_drm modeset` / `fbdev` | Whether the run is attributable to the required state | `Y` / `N` |
| `unified addressing (UVA)` | Whether the device pointer can equal the host pointer | PASS; without it the technique cannot work |
| `allocator returned a pool` | Whether a 1.75 GiB DRM dumb buffer was accepted and registered | PASS |
| `MemAvailable delta` | **The central claim.** Whether the 1.75 GiB came out of ordinary RAM | Delta ≈ ordinary prefix only, **not** ≈ 1.75 GiB |
| `device write + readback (display)` | Whether CUDA can actually use the region | PASS |
| `cuMemGetInfo total unchanged` | Whether this is ordinary `cudaMalloc` budget | PASS — it is *not* |
| `read penalty` | What the region costs | Some penalty; the author measured one |

Two outcomes are both informative:

- **`allocation FAILED`** at the `DRM create scanout` step means the driver rejected a
  4096 × 114688 dumb buffer. This is the most likely way the technique fails on our
  driver, and it is a *result*, not a mistake: record the error text. The remedy to try
  next is splitting the region into several smaller dumb buffers mapped back-to-back with
  `MAP_FIXED` into the same reserved span, and registering each mapping separately.
- **`MemAvailable delta` ≈ 1.75 GiB** means the region came out of ordinary RAM, i.e. the
  dumb buffer was *not* served from the display reservation. The unlock would then be worth
  nothing, and stage 5 must not be attempted.

---

## Stage 4 — negative control

A positive result is only credible with a negative control. On the same node:

```bash
# Reload the module the other way and reboot.
sudo rm -f /etc/modprobe.d/ds41-display-kv.conf
sudo update-initramfs -u -k all
sudo reboot

sudo cat /sys/module/nvidia_drm/parameters/fbdev      # expect Y now
./gb10_display_probe --ordinary-mib 0
```

Expected: either the dumb-buffer allocation fails, or the region is not available in the
same way. Record what actually happens; if the result is identical with `fbdev=Y`, then
`fbdev=0` is not what makes the difference and the mechanism is not what it claims.

Then restore stage 1 before any further work.

---

## Stage 5 — only if stages 3 and 4 pass: port into our stack

Outline only, deliberately not specified in detail until stages 0–4 pass. Do not start this
on the fleet's serving nodes before the single-node measurement is understood.

The portable parts and the parts that need rework:

| Piece | Portable? |
| --- | --- |
| `display_kv.c` allocator, unmodified | Yes — it only needs `/dev/dri/cardN` and libcuda |
| Host preparation (stage 1) | Yes, per node |
| The `cuMemHostRegister(IOMEMORY)` idea | Yes, and it is the documented API for device memory |
| `display_kv.py` integration | **No** — it patches an exact vLLM module by SHA-256, which our build does not have |
| KV budget/accounting changes | **No** — recipe-specific (`ordinary`/`external` splits, 0.92, six slots) |

Work items for our stack, in order:

1. Find where our vLLM build allocates KV cache tensors. The idea to reuse: wrap the backing
   allocation so the KV tensor is a slice of the external span, not of a `torch.zeros` buffer.
2. Decide how much of the 1.75 GiB to claim. The author claims the whole 1.75 GiB with zero
   ordinary KV; claiming less, and keeping ordinary KV, is a safer first test.
3. Expose the DRM node to the serving container as the author does: `--device=/dev/dri/card0`,
   `--group-add <gid of /dev/dri/card0>`, and `NVIDIA_DRIVER_CAPABILITIES` including
   `display`. Not a privileged container.
4. Check interaction with our `--gpu-memory-utilization` admission. This is the main risk: an
   external KV backing is invisible to the ordinary accounting, and our startup admission checks
   must still be satisfied. This is where our build can differ from the author's.
5. Keep the allocation alive for the lifetime of the cache and never release it while tensors or
   CUDA graphs are live — the author's integration treats that as a failure condition.

Measure on one node before rolling to four: the gain in KV tokens at our target context length,
against the same configuration without the display backing.

---

## Stage 6 — rollback

```bash
# Stop all serving and CUDA work on the node first.
sudo rm -f /etc/modprobe.d/ds41-display-kv.conf
sudo systemctl set-default graphical.target        # or the value saved in stage 1
sudo update-initramfs -u -k all
sudo reboot
```

Then confirm the stage 0 values are back. Remove only the file this procedure created; do not
remove unrelated NVIDIA configuration.

---

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `open DRM card: Permission denied` | `ls -l /dev/dri/card0`, then `id`. Add the user to the owning group and reconnect. Do not run as root. |
| `open DRM card: No such file or directory` | `nvidia_drm` is not loaded or the node has no NVIDIA DRM card. Check `modeset` and the card inventory from stage 0f. |
| `modprobe: unknown parameter 'fbdev'` | The driver was built without framebuffer-device support. Check `/sys/module/nvidia_drm/parameters/` for the parameters that exist, and re-evaluate whether the module state is even the relevant variable here. |
| Parameter files unreadable as an ordinary user after reboot | Recorded in issue #1: `update-initramfs -u -k all` did not keep `644` permissions. Read with `sudo` or add a udev rule. |
| `DRM create scanout` fails | Dumb-buffer geometry (4096 × 114688) rejected by the driver; try splitting the span into several smaller buffers mapped with `MAP_FIXED`, and record the driver error. |
| `Unexpected scanout size` | The driver did not return the requested size; the geometry arithmetic in §3 of the analysis did not hold on this driver. |
| `Driver did not preserve contiguous UVA` | UVA is not available or the device pointer differs from the host pointer. The technique cannot work on this node as written. |
| `register display IO` fails | `cuMemHostRegister` refused the mapping. This is the CUDA-side gate; check the driver version against the qualified 580.173.02. |
| Registering works but `MemAvailable` also drops 1.75 GiB | The buffer came from ordinary RAM, not the reservation. The unlock is worth nothing here. |
| `rmmod nvidia_drm` fails | Something holds the module (a desktop or CUDA job) or `refcnt` is not 0. Stop that workload; do not force the unload. |
| Desktop gone and cannot reconnect | Stage 1 was applied without confirming SSH first. Recover at the console and restore the boot target from the stage 1 record. |
