#!/usr/bin/env bash
# ds41fp4-build-image.sh: build vlspeed-eng:5 on all four Sparks.
#
# vlspeed-eng:5 = vlspeed-eng:4 (see vlspeed-build-image.sh)
#   + engram-fp4-disk.py -> packed E2M1 (FP4) Engram rows for the fleet's
#     MXFP4+FP4-Engram hybrid checkpoint.
#
# The upstream stack's disk Engram reader stages fp8_e4m3fn[256] + ue8m0[8]
# rows. The hybrid stores the same table as uint8[rows, 128] packed E2M1 with
# the same-shaped E8M0 scale plane, so an unchanged boot dies at weight load.
# This layer is Python only; nothing is compiled here, so the build is about a
# second per box. See docs/INTEGRATION.md.
#
# Everything upstream stays pristine: the three scripts above are verbatim, and
# this layer applies patch/engram-fp4-disk.py on top of the installed tree, the
# same overlay pattern the ds4.1 stack uses for its FP4 disk adapter.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
WORK="${WORK:-$HOME/cc-scratch/vlspeed}"
BOXES=(192.168.0.1 192.168.0.2 192.168.0.3 192.168.0.4)
SSH_OPTS="-o StrictHostKeyChecking=no -o BatchMode=yes"

VLLM_ROOT=/usr/local/lib/python3.12/dist-packages/vllm
DISK_PY=$VLLM_ROOT/models/deepseek_v4_1/common/engram_disk.py
ENGRAM_PY=$VLLM_ROOT/models/deepseek_v4_1/common/engram.py

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
cp "$HERE/../patch/engram-fp4-disk.py" "$tmp/"
cat > "$tmp/Dockerfile" <<DOCK
FROM vlspeed-eng:4
COPY engram-fp4-disk.py /opt/vl41/
RUN python3 /opt/vl41/engram-fp4-disk.py $DISK_PY $ENGRAM_PY
DOCK
tar czf "$tmp/ctx.tgz" -C "$tmp" engram-fp4-disk.py Dockerfile

for h in "${BOXES[@]}"; do
  echo ">>> $h"
  ssh $SSH_OPTS "$h" "mkdir -p $WORK"
  scp -q $SSH_OPTS "$tmp/ctx.tgz" "$h:$WORK/"
  ssh $SSH_OPTS "$h" "rm -rf $WORK/b && mkdir -p $WORK/b && cd $WORK/b && tar xzf ../ctx.tgz && docker build -t vlspeed-eng:5 . | tail -3"
done
