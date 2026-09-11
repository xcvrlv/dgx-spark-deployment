#!/usr/bin/env bash
# Per-host control for the bounded DS41 page-cache flusher. Installed next to
# cache-flusher.sh and invoked over SSH by cluster.py's flush-cache action.
# usage: cache-flusher-remote.sh <start|stop|status> <duration> <script_path> <state_dir>
# Paths must be absolute where used; all DS41 cluster paths are identical on peers.
set -euo pipefail

action="${1:-}"
duration="${2:-5400}"
script="${3:-}"
state="${4:-}"

usage() {
  echo "usage: $0 <start|stop|status> <duration> <script_path> <state_dir>" >&2
}

case "$action" in
  start|stop|status) ;;
  *) usage; exit 2 ;;
esac
case "$duration" in
  ''|*[!0-9]*|0) echo "duration must be a positive integer" >&2; exit 2 ;;
esac
case "$state" in
  ''|/*) ;;
  *) usage; echo "state_dir must be absolute" >&2; exit 2 ;;
esac
# Only start runs the daemon; status/stop need the state dir alone.
if [[ "$action" == "start" ]]; then
  case "$script" in
    ''|/*) ;;
    *) usage; echo "script_path must be absolute" >&2; exit 2 ;;
  esac
fi

pid_file="$state/pid"
log_file="$state/flusher.log"

case "$action" in
  start)
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
    ;;
  status)
    if [[ -s "$pid_file" ]]; then
      pid="$(cat "$pid_file")"
      if kill -0 "$pid" 2>/dev/null; then
        echo "active pid=$pid"
        exit 0
      fi
    fi
    echo "inactive"
    exit 1
    ;;
  stop)
    if [[ ! -s "$pid_file" ]]; then
      echo "already stopped"
      exit 0
    fi
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      # setsid makes the flusher PID its process-group ID. Signal the group so
      # an in-flight sleep exits immediately instead of delaying shutdown by 60s.
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
    ;;
esac
