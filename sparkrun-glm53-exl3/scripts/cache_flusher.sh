#!/usr/bin/env bash
# Keep clean file-backed pages out of the GB10 unified-memory pool while vLLM
# loads. This must run on the host, not in the unprivileged serving container.
#
# The loop is intentionally unconditional. A Cached/MemAvailable threshold can
# leave several GiB resident and still make CUDA's startup admission check fail.
set -euo pipefail

duration="${CACHE_FLUSHER_DURATION_SECONDS:-${1:-5400}}"
interval="${CACHE_FLUSHER_INTERVAL_SECONDS:-60}"
lock_file="${CACHE_FLUSHER_LOCK_FILE:-${XDG_RUNTIME_DIR:-/tmp}/glm53-cache-flusher.lock}"

case "$duration" in
  ''|*[!0-9]*|0) echo "FATAL: duration must be a positive integer (seconds)" >&2; exit 2 ;;
esac
case "$interval" in
  ''|*[!0-9]*|0) echo "FATAL: CACHE_FLUSHER_INTERVAL_SECONDS must be a positive integer" >&2; exit 2 ;;
esac

command -v flock >/dev/null 2>&1 || {
  echo "FATAL: flock is required to prevent duplicate cache flushers" >&2
  exit 1
}

exec 9>"$lock_file"
if ! flock -n 9; then
  echo "FATAL: another GLM-5.3 cache flusher already holds $lock_file" >&2
  exit 1
fi

stopping=0
on_stop() {
  stopping=1
}
trap on_stop INT TERM

drop_page_cache() {
  # sync runs without privilege. The only privileged operation is writing the
  # fixed drop_caches path, matching SparkRun's scoped clear-cache sudo rule.
  sync
  if ! printf '3\n' | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; then
    echo "FATAL: cannot drop page cache with non-interactive sudo" >&2
    echo "Run: sparkrun setup clear-cache --hosts <all-hosts> --save-sudo" >&2
    return 1
  fi
  printf 'flusher: dropped page cache at %s\n' "$(date --iso-8601=seconds)"
}

# Prove the exact privileged operation works before advertising readiness. The
# cluster launcher waits for the line below and refuses to start vLLM without it.
drop_page_cache
echo "flusher: starting, unconditional, every ${interval}s for ${duration}s (pid $$)"

deadline=$((SECONDS + duration))
while ((SECONDS < deadline)) && ((stopping == 0)); do
  remaining=$((deadline - SECONDS))
  sleep_for="$interval"
  ((remaining < sleep_for)) && sleep_for="$remaining"
  sleep "$sleep_for" || true
  ((stopping == 0)) || break
  ((SECONDS < deadline)) || break
  drop_page_cache
done

if ((stopping == 1)); then
  echo "flusher: stopped"
else
  echo "flusher: window elapsed after ${duration}s; restart it if vLLM is still loading"
fi
