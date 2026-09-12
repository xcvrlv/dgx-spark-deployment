#!/bin/bash
# distinct-content prompts of one size, seeds given; guard log memguard8.log
SP=${SP:-/tmp}
LOG=${MEMGUARD_LOG:-$SP/memguard.log}; size=$1; shift
mem() { awk '/MemAvailable/{printf "%.2f",$2/1048576}' /proc/meminfo; }
for seed in "$@"; do
  s=$(date +%H:%M:%S); before=$(mem)
  out=$(ssh -i ~/.ssh/id_ed25519_shared zurih@10.0.0.2 "python3 /tmp/prefill_distinct.py $size $seed" 2>&1 | tail -1)
  e=$(date +%H:%M:%S); sleep 3; after=$(mem)
  min=$(awk -v s="$s" -v e="$e" '$1>=s && $1<=e && $2+0>0 {print $2}' $LOG | sort -n | head -1)
  aborts=$(awk -v s="$s" -v e="$e" '$1>=s && $1<=e && /ABORT/' $LOG | wc -l)
  echo "RAMP2 size=$size seed=$seed: before=${before} GB min=${min} GB after=${after} GB aborts=${aborts} | $out"
  if [ "$aborts" != "0" ]; then echo "RAMP2 STOP (guard fired)"; exit 1; fi
done
