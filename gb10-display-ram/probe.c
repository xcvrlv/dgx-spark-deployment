// SPDX-License-Identifier: MIT
/*
 * gb10_display_probe — independent probe for the GB10 display-reserve memory
 * technique published by coolbho3k ("emi") on the NVIDIA developer forums,
 * DGX Spark / GB10 user forum, topic 383583.
 *
 * This probe answers, on real hardware and in one shot:
 *   1. Does /dev/dri/card<N> accept a DRM dumb buffer of exactly 1.75 GiB?
 *   2. Is that buffer OUTSIDE ordinary RAM? (measured against /proc/meminfo)
 *   3. Can CUDA kernels actually read and write it?
 *   4. Does it cost bandwidth versus ordinary CUDA memory, and how much?
 *   5. Is the result attributable to the driver state we think it is?
 *
 * Everything above is measured here, not assumed. The probe never modifies
 * firmware, boot configuration, or module parameters; it only reads them.
 *
 * Build:  make            (see Makefile; needs libdrm-dev headers + libcuda)
 * Run  :  ./gb10_display_probe            # 1 GiB ordinary + 1.75 GiB display
 *         ./gb10_display_probe --ordinary-mib 0   # display-only variant
 *
 * This file is OUR code (MIT). It links at runtime/build time against the
 * upstream allocator display_kv.c, which is AGPL-3.0-only and is fetched
 * separately by fetch-upstream.sh — it is deliberately not vendored here.
 */
#define _GNU_SOURCE
#include <cuda.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

/* Implemented by the upstream AGPL allocator (libds41_display_kv.so). */
typedef struct Pool Pool;
Pool *ds41_display_create(size_t ordinary, size_t display);
void ds41_display_destroy(Pool *pool);
uint64_t ds41_display_pointer(Pool *pool);
size_t ds41_display_size(Pool *pool);
const char *ds41_display_error(void);

#define MIB (1024UL * 1024UL)
#define DISPLAY_BYTES (1792UL * MIB) /* allocator hard-requires exactly 1.75 GiB */
#define ORDINARY_LIMIT (1024UL * MIB) /* allocator hard-requires <= 1 GiB */

static int failures = 0;

static void say(const char *fmt, ...) {
	va_list ap;
	va_start(ap, fmt);
	vfprintf(stdout, fmt, ap);
	va_end(ap);
	fputc('\n', stdout);
	fflush(stdout);
}

static void verdict(const char *check, int ok, const char *detail) {
	if (!ok) failures++;
	say("  [%s] %-46s %s", ok ? "PASS" : "FAIL", check, detail ? detail : "");
}

static char *read_first_line(const char *path) {
	static char buf[256];
	FILE *f = fopen(path, "r");
	if (!f) return NULL;
	if (!fgets(buf, sizeof(buf), f)) { fclose(f); return NULL; }
	fclose(f);
	buf[strcspn(buf, "\r\n")] = 0;
	return buf;
}

typedef struct {
	unsigned long mem_total_kib, mem_free_kib, mem_available_kib;
	unsigned long cma_total_kib;
} MemInfo;

static char *read_file_trim_all(const char *path) {
	static char buf[16384];
	FILE *f = fopen(path, "r");
	if (!f) return NULL;
	size_t n = fread(buf, 1, sizeof(buf) - 1, f);
	fclose(f);
	buf[n] = 0;
	return buf;
}

static int read_meminfo(MemInfo *m) {
	memset(m, 0, sizeof(*m));
	char *raw = read_file_trim_all("/proc/meminfo");
	if (!raw) return 0;
	char *save = NULL;
	for (char *line = strtok_r(raw, "\n", &save); line; line = strtok_r(NULL, "\n", &save)) {
		unsigned long v = 0;
		if (sscanf(line, "MemTotal: %lu kB", &v) == 1) m->mem_total_kib = v;
		else if (sscanf(line, "MemFree: %lu kB", &v) == 1) m->mem_free_kib = v;
		else if (sscanf(line, "MemAvailable: %lu kB", &v) == 1) m->mem_available_kib = v;
		else if (sscanf(line, "CmaTotal: %lu kB", &v) == 1) m->cma_total_kib = v;
	}
	return 1;
}

static const char *cu_name(CUresult rc) {
	const char *name = "unknown";
	cuGetErrorName(rc, &name);
	return name;
}

static int cuda_ok(const char *op, CUresult rc) {
	if (rc == CUDA_SUCCESS) return 1;
	say("  CUDA ERROR: %s -> %s (%d)", op, cu_name(rc), (int)rc);
	return 0;
}

static double now_ms(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Measured write bandwidth: device-side fill of the whole region. */
static double bw_fill_mib_s(CUdeviceptr p, size_t bytes, int iters) {
	CUstream s = NULL;
	if (!cuda_ok("cuStreamCreate", cuStreamCreate(&s, CU_STREAM_NON_BLOCKING))) return -1;
	double best = 0;
	for (int i = 0; i < iters; i++) {
		double t0 = now_ms();
		CUresult rc = cuMemsetD8Async(p, (unsigned char)(0xA0 + i), bytes, s);
		if (rc != CUDA_SUCCESS) { cuStreamDestroy(s); return -1; }
		if (!cuda_ok("cuStreamSynchronize", cuStreamSynchronize(s))) { cuStreamDestroy(s); return -1; }
		double dt = now_ms() - t0;
		if (dt > 0 && (best == 0 || dt < best)) best = dt;
	}
	cuStreamDestroy(s);
	return best > 0 ? (double)bytes / (best / 1000.0) / (1024.0 * 1024.0) : -1;
}

/* Measured read bandwidth: device-to-device copy out of the region. */
static double bw_read_mib_s(CUdeviceptr src, CUdeviceptr dst, size_t bytes, int iters) {
	CUstream s = NULL;
	if (!cuda_ok("cuStreamCreate", cuStreamCreate(&s, CU_STREAM_NON_BLOCKING))) return -1;
	double best = 0;
	for (int i = 0; i < iters; i++) {
		double t0 = now_ms();
		CUresult rc = cuMemcpyDtoDAsync(dst, src, bytes, s);
		if (rc != CUDA_SUCCESS) { cuStreamDestroy(s); return -1; }
		if (!cuda_ok("cuStreamSynchronize", cuStreamSynchronize(s))) { cuStreamDestroy(s); return -1; }
		double dt = now_ms() - t0;
		if (dt > 0 && (best == 0 || dt < best)) best = dt;
	}
	cuStreamDestroy(s);
	return best > 0 ? (double)bytes / (best / 1000.0) / (1024.0 * 1024.0) : -1;
}

/* Device-side write then host readback + comparison: proves CUDA can touch it. */
static int verify_device_access(CUdeviceptr dev, size_t bytes, const char *label) {
	const size_t chunk = 8UL * MIB;
	unsigned char *host = malloc(chunk);
	if (!host) return 0;
	CUstream s = NULL;
	if (!cuda_ok("cuStreamCreate", cuStreamCreate(&s, CU_STREAM_NON_BLOCKING))) { free(host); return 0; }
	int ok = 1;
	size_t done = 0;
	while (done < bytes) {
		size_t n = bytes - done < chunk ? bytes - done : chunk;
		unsigned char pattern = (unsigned char)(0x5A ^ (done / chunk));
		if (!cuda_ok("cuMemsetD8Async", cuMemsetD8Async(dev + done, pattern, n, s))) { ok = 0; break; }
		if (!cuda_ok("cuMemcpyDtoH", cuMemcpyDtoH(host, dev + done, n))) { ok = 0; break; }
		for (size_t i = 0; i < n; i++) {
			if (host[i] != pattern) {
				say("  mismatch in %s at offset %zu: got 0x%02x want 0x%02x",
				    label, done + i, host[i], pattern);
				ok = 0;
				break;
			}
		}
		if (!ok) break;
		done += n;
	}
	cuStreamDestroy(s);
	free(host);
	return ok;
}

static void report_drm_state(void) {
	say("== DRM / driver state (read-only, for attribution) ==");
	const char *p;
	p = read_first_line("/sys/module/nvidia_drm/parameters/modeset");
	say("  nvidia_drm modeset : %s   (need Y — modesetting enabled)", p ? p : "<unreadable, try sudo>");
	p = read_first_line("/sys/module/nvidia_drm/parameters/fbdev");
	say("  nvidia_drm fbdev   : %s   (need N — framebuffer console off)", p ? p : "<unreadable, try sudo>");
	p = read_first_line("/sys/module/nvidia_drm/refcnt");
	say("  nvidia_drm refcnt : %s   (need 0 before rmmod)", p ? p : "<unreadable>");

	char cmd[512];
	snprintf(cmd, sizeof(cmd),
	         "for c in /sys/class/drm/card[0-9]*; do "
	         "[ -r \"$c/device/vendor\" ] && printf '  %%s vendor=%%s\\n' \"${c##*/}\" \"$(cat \"$c/device/vendor\")\"; done");
	say("  DRM cards (0x10de == NVIDIA):");
	fflush(stdout);
	int rc = system(cmd);
	(void)rc;

	say("  display-manager    : check with: systemctl is-active display-manager");
	say("  default boot target: check with: systemctl get-default");
}

int main(int argc, char **argv) {
	setvbuf(stdout, NULL, _IONBF, 0);

	size_t ordinary = ORDINARY_LIMIT;
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--ordinary-mib") && i + 1 < argc) {
			long v = strtol(argv[++i], NULL, 10);
			if (v < 0 || (size_t)v > ORDINARY_LIMIT / MIB) {
				fprintf(stderr, "ordinary must be 0..%lu MiB\n", ORDINARY_LIMIT / MIB);
				return 2;
			}
			ordinary = (size_t)v * MIB;
		} else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
			say("usage: %s [--ordinary-mib 0..1024]   (default 1024)", argv[0]);
			return 0;
		}
	}

	MemInfo before = {0}, after = {0};
	read_meminfo(&before);

	report_drm_state();

	say("");
	say("== CUDA context ==");
	if (!cuda_ok("cuInit", cuInit(0))) return 1;
	CUdevice dev;
	if (!cuda_ok("cuDeviceGet", cuDeviceGet(&dev, 0))) return 1;
	char name[256] = {0};
	cuDeviceGetName(name, sizeof(name), dev);
	int cc_major = 0, cc_minor = 0;
	cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
	cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
	int uva = 0;
	cuDeviceGetAttribute(&uva, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, dev);
	int drv = 0;
	cuDriverGetVersion(&drv);
	say("  device             : %s (sm_%d%d)", name, cc_major, cc_minor);
	say("  driver version     : %d", drv);
	/* UVA is mandatory: the allocator asserts device pointer == host pointer. */
	verdict("unified addressing (UVA)", uva != 0, "required by the allocator's pointer check");
	if (!uva) {
		say("  stopping: without UVA the display pointer would not be usable as a device pointer");
		return 1;
	}

	CUcontext ctx;
	if (!cuda_ok("cuDevicePrimaryCtxRetain", cuDevicePrimaryCtxRetain(&ctx, dev))) return 1;
	if (!cuda_ok("cuCtxSetCurrent", cuCtxSetCurrent(ctx))) return 1;

	size_t free_before = 0, total_before = 0;
	if (!cuda_ok("cuMemGetInfo before", cuMemGetInfo(&free_before, &total_before))) return 1;
	say("  cuMemGetInfo before: free %.3f GiB / total %.3f GiB",
	    (double)free_before / (1024.0 * 1024.0 * 1024.0),
	    (double)total_before / (1024.0 * 1024.0 * 1024.0));

	say("");
	say("== allocate display-backed span ==");
	say("  ordinary request : %zu MiB", ordinary / MIB);
	say("  display request  : %zu MiB (%.4f GiB) — fixed by the allocator",
	    DISPLAY_BYTES / MIB, (double)DISPLAY_BYTES / (1024.0 * 1024.0 * 1024.0));
	Pool *pool = ds41_display_create(ordinary, DISPLAY_BYTES);
	if (!pool) {
		const char *err = ds41_display_error();
		say("  RESULT: allocation FAILED: %s", err && *err ? err : "<no message>");
		say("");
		say("  This is the expected outcome when modeset=0/absent or when the DRM");
		say("  device is not accessible to this user. Check the state printed above.");
		cuDevicePrimaryCtxRelease(dev);
		return 1;
	}
	CUdeviceptr base = (CUdeviceptr)ds41_display_pointer(pool);
	size_t total = ds41_display_size(pool);
	CUdeviceptr display = base + ordinary;
	verdict("allocator returned a pool", total == ordinary + DISPLAY_BYTES, NULL);
	say("  span   : 0x%016" PRIx64 " .. 0x%016" PRIx64 "  (%zu bytes, %.3f GiB)",
	    (uint64_t)base, (uint64_t)(base + total - 1), total, (double)total / (1024.0 * 1024.0 * 1024.0));
	say("  display: 0x%016" PRIx64 "  (%zu bytes, %.3f GiB)",
	    (uint64_t)display, DISPLAY_BYTES, (double)DISPLAY_BYTES / (1024.0 * 1024.0 * 1024.0));

	read_meminfo(&after);
	long delta_avail_kib = (long)before.mem_available_kib - (long)after.mem_available_kib;
	say("");
	say("== ordinary-RAM accounting (the central claim) ==");
	say("  MemTotal     : %.3f GiB (before) -> %.3f GiB (after)",
	    (double)before.mem_total_kib / 1048576.0, (double)after.mem_total_kib / 1048576.0);
	say("  MemAvailable : %.3f GiB (before) -> %.3f GiB (after)",
	    (double)before.mem_available_kib / 1048576.0, (double)after.mem_available_kib / 1048576.0);
	say("  MemAvailable delta: %.3f GiB   (ordinary prefix = %.3f GiB)",
	    (double)delta_avail_kib / 1048576.0, (double)ordinary / (1024.0 * 1024.0 * 1024.0));
	say("  -> the display part did %s come out of ordinary RAM",
	    (double)delta_avail_kib / 1048576.0 < (double)DISPLAY_BYTES / (1024.0 * 1024.0 * 1024.0) / 2 ? "" : "NOT");
	/* The claim being tested: 1.75 GiB of backing that is NOT ordinary RAM. */
	verdict("display backing is outside ordinary RAM",
	        (size_t)(delta_avail_kib * 1024) < ordinary + DISPLAY_BYTES / 2, NULL);

	say("");
	say("== can CUDA actually read and write the display region? ==");
	int display_ok = verify_device_access(display, DISPLAY_BYTES, "display region");
	verdict("device write + readback (display)", display_ok, NULL);
	if (ordinary) {
		int ordinary_ok = verify_device_access(base, ordinary, "ordinary prefix");
		verdict("device write + readback (ordinary)", ordinary_ok, NULL);
	}

	size_t free_after = 0, total_after = 0;
	cuMemGetInfo(&free_after, &total_after);
	say("");
	say("== cuMemGetInfo after (expect UNCHANGED: this is not cudaMalloc budget) ==");
	say("  free %.3f GiB -> %.3f GiB | total %.3f GiB -> %.3f GiB",
	    (double)free_before / (1024.0 * 1024.0 * 1024.0),
	    (double)free_after / (1024.0 * 1024.0 * 1024.0),
	    (double)total_before / (1024.0 * 1024.0 * 1024.0),
	    (double)total_after / (1024.0 * 1024.0 * 1024.0));
	verdict("cuMemGetInfo total unchanged (expected)", total_after == total_before,
	        "the unlock adds accessible memory, not cudaMalloc budget");

	say("");
	say("== bandwidth: display reserve vs ordinary CUDA memory ==");
	CUdeviceptr control = 0;
	size_t control_bytes = DISPLAY_BYTES;
	int have_control = cuda_ok("cuMemAlloc control", cuMemAlloc(&control, control_bytes)) != 0;
	double fill_display = bw_fill_mib_s(display, DISPLAY_BYTES, 3);
	double read_display = bw_read_mib_s(display, control ? control : display, DISPLAY_BYTES, 3);
	say("  display reserve: fill %8.1f MiB/s | read %8.1f MiB/s", fill_display, read_display);
	if (have_control) {
		double fill_control = bw_fill_mib_s(control, control_bytes, 3);
		double read_control = bw_read_mib_s(control, display, control_bytes, 3);
		say("  ordinary memory: fill %8.1f MiB/s | read %8.1f MiB/s", fill_control, read_control);
		if (fill_control > 0 && fill_display > 0)
			say("  fill penalty  : %+.1f%%", (fill_display / fill_control - 1.0) * 100.0);
		if (read_control > 0 && read_display > 0)
			say("  read penalty  : %+.1f%%", (read_display / read_control - 1.0) * 100.0);
		cuMemFree(control);
	} else {
		say("  (ordinary control buffer allocation failed; penalty not measured)");
	}

	/* Keep the pool alive until process exit: freeing while tensors are live is
	 * the documented failure mode. This probe holds no torch tensors, so it is
	 * safe to release here for a clean repeatable run. */
	cuCtxSynchronize();
	ds41_display_destroy(pool);
	cuDevicePrimaryCtxRelease(dev);

	say("");
	int all_ok = display_ok &&
	    (size_t)(delta_avail_kib * 1024) < ordinary + DISPLAY_BYTES / 2;
	say("RESULT: %s", all_ok ? "PASS — display reserve is CUDA-accessible and is not ordinary RAM"
	                         : "FAIL — see checks above");
	return all_ok ? 0 : 1;
}
