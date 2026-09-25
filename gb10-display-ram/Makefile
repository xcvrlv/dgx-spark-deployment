# Build the GB10 display-reserve probe.
#
# Produces:
#   upstream/libds41_display_kv.so — the upstream AGPL-3.0-only allocator.
#       Fetched prebuilt (aarch64) by fetch-upstream.sh, or built from the
#       pinned source with `make libds41_display_kv.so`.
#   gb10_display_probe           — this repository's probe, linked against the
#       allocator plus libcuda.
#
# Requirements on the Spark (or inside the serving container):
#   libdrm-dev      provides <drm/drm.h>   (apt-get install -y libdrm-dev)
#   libcuda.so.1     provides the CUDA driver API (from the NVIDIA driver)
#   build-essential  provides cc
#
# If `-lcuda` fails because only libcuda.so.1 exists (typical inside the
# NVIDIA container), build with:  make CUDA_LIB='-l:libcuda.so.1'

CC       ?= cc
CFLAGS   ?= -O2 -g -Wall -Wextra -std=gnu11
CUDA_LIB ?= -lcuda

UPSTREAM_DIR := upstream
ALLOC_SO     := $(UPSTREAM_DIR)/libds41_display_kv.so
ALLOC_SRC    := $(UPSTREAM_DIR)/display_kv.c

.PHONY: all fetch check clean

all: gb10_display_probe

fetch:
	./fetch-upstream.sh

# Prefer the fetched prebuilt library; build from source only if it is absent.
$(ALLOC_SO):
	@if [ -f "$@" ]; then \
		echo "using fetched prebuilt $@"; \
	elif [ -f "$(ALLOC_SRC)" ]; then \
		echo "building $@ from pinned source (AGPL-3.0-only)"; \
		$(CC) $(CFLAGS) -fPIC -shared -o $@ $(ALLOC_SRC); \
	else \
		echo "missing $(ALLOC_SO); run ./fetch-upstream.sh first" >&2; \
		exit 1; \
	fi

gb10_display_probe: probe.c $(ALLOC_SO)
	$(CC) $(CFLAGS) -o $@ probe.c -L$(UPSTREAM_DIR) -lds41_display_kv $(CUDA_LIB) \
	    -Wl,-rpath,'$$ORIGIN/$(UPSTREAM_DIR)'

check: all
	@echo "== built =="
	@ls -l gb10_display_probe $(ALLOC_SO)
	@echo
	@echo "== allocator exports (expect ds41_display_create/destroy/pointer/size/error) =="
	@nm -D --defined-only $(ALLOC_SO) 2>/dev/null | grep ds41_display || true
	@echo
	@echo "== probe links libcuda.so.1 only =="
	@ldd gb10_display_probe 2>/dev/null | grep -Ei 'cuda|drm' || true

clean:
	rm -f gb10_display_probe
