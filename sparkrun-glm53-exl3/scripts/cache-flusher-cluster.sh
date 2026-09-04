#!/usr/bin/env bash
# Install and manage the bounded GLM-5.3 page-cache flusher on a host list.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
flusher="$root_dir/scripts/cache_flusher.sh"
action="${1:-}"
hosts_csv="${2:-}"
duration="${CACHE_FLUSHER_DURATION_SECONDS:-5400}"
remote_dir='.local/libexec/glm53-cache-flusher'
remote_script="$remote_dir/cache_flusher.sh"
state_dir='.cache/sparkrun/glm53-cache-flusher'

usage() {
  echo "usage: $0 <start|status|stop> <host1,host2,...>" >&2
}

case "$action" in
  start|status|stop) ;;
  *) usage; exit 2 ;;
esac
[[ -n "$hosts_csv" ]] || { usage; exit 2; }
case "$duration" in
  ''|*[!0-9]*|0) echo "duration must be a positive integer" >&2; exit 2 ;;
esac

IFS=',' read -r -a hosts <<<"$hosts_csv"
((${#hosts[@]} > 0)) || { usage; exit 2; }

ssh_options=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
scp_options=(-q -o BatchMode=yes -o ConnectTimeout=10)
if [[ -n "${SPARK_SSH_KEY:-}" ]]; then
  ssh_options+=(-i "$SPARK_SSH_KEY")
  scp_options+=(-i "$SPARK_SSH_KEY")
fi

validate_host() {
  local host="$1"
  [[ "$host" =~ ^[A-Za-z0-9_.:@-]+$ ]] || {
    echo "invalid SSH host: $host" >&2
    return 1
  }
}

install_on() {
  local host="$1" remote_tmp="$remote_dir/cache_flusher.sh.tmp.$$"
  ssh "${ssh_options[@]}" "$host" \
    "mkdir -p \"\$HOME/$remote_dir\" \"\$HOME/$state_dir\""
  scp "${scp_options[@]}" "$flusher" "$host:$remote_tmp"
  ssh "${ssh_options[@]}" "$host" \
    "chmod 0755 \"\$HOME/$remote_tmp\" && mv -f \"\$HOME/$remote_tmp\" \"\$HOME/$remote_script\""
}

start_on() {
  local host="$1"
  ssh "${ssh_options[@]}" "$host" "bash -s" -- "$duration" "$remote_script" "$state_dir" <<'REMOTE'
set -euo pipefail
duration="$1"
script="$HOME/$2"
state="$HOME/$3"
pid_file="$state/pid"
log_file="$state/flusher.log"

mkdir -p "$state"
if [[ -s "$pid_file" ]]; then
  pid="$(cat "$pid_file")"
  if kill -0 "$pid" 2>/dev/null; then
    # Refresh the full load window on every launch. Reusing a nearly-expired
    # process could leave a long cold load unprotected after preflight passed.
    echo "flusher: restarting active pid $pid to refresh the load window"
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
    for _ in $(seq 1 10); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      echo "FATAL: existing flusher pid $pid did not stop" >&2
      exit 1
    fi
  fi
  rm -f "$pid_file"
fi

: >"$log_file"
setsid nohup "$script" "$duration" >"$log_file" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" >"$pid_file"

for _ in $(seq 1 60); do
  if grep -q 'flusher: starting, unconditional' "$log_file" && kill -0 "$pid" 2>/dev/null; then
    echo "flusher: active (pid $pid)"
    exit 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    break
  fi
  sleep 1
done

echo "FATAL: cache flusher did not become ready" >&2
sed -n '1,40p' "$log_file" >&2
if kill -0 "$pid" 2>/dev/null; then
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
fi
rm -f "$pid_file"
exit 1
REMOTE
}

status_on() {
  local host="$1"
  ssh "${ssh_options[@]}" "$host" "bash -s" -- "$state_dir" <<'REMOTE'
set -euo pipefail
state="$HOME/$1"
pid_file="$state/pid"
if [[ -s "$pid_file" ]]; then
  pid="$(cat "$pid_file")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "active pid=$pid"
    exit 0
  fi
fi
echo "inactive"
exit 1
REMOTE
}

stop_on() {
  local host="$1"
  ssh "${ssh_options[@]}" "$host" "bash -s" -- "$state_dir" <<'REMOTE'
set -euo pipefail
state="$HOME/$1"
pid_file="$state/pid"
if [[ ! -s "$pid_file" ]]; then
  echo "already stopped"
  exit 0
fi
pid="$(cat "$pid_file")"
if kill -0 "$pid" 2>/dev/null; then
  # setsid makes the flusher PID its process-group ID. Signal the group so an
  # in-flight sleep exits immediately instead of delaying shutdown by 60s.
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
  for _ in $(seq 1 10); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "FATAL: flusher pid $pid did not stop" >&2
    exit 1
  fi
fi
rm -f "$pid_file"
echo "stopped"
REMOTE
}

for host in "${hosts[@]}"; do
  validate_host "$host"
done

case "$action" in
  start)
    started=()
    for host in "${hosts[@]}"; do
      echo "[$host] installing cache flusher"
      if install_on "$host"; then
        echo "[$host] starting cache flusher"
        if start_on "$host"; then
          started+=("$host")
          continue
        fi
      fi
      echo "cache flusher failed on $host; stopping the nodes already started" >&2
      for started_host in "${started[@]}"; do
        stop_on "$started_host" || true
      done
      exit 1
    done
    ;;
  status)
    failed=0
    for host in "${hosts[@]}"; do
      printf '[%s] ' "$host"
      status_on "$host" || failed=1
    done
    exit "$failed"
    ;;
  stop)
    failed=0
    for host in "${hosts[@]}"; do
      printf '[%s] ' "$host"
      stop_on "$host" || failed=1
    done
    exit "$failed"
    ;;
esac
