#!/usr/bin/env bash
# Hand report for every policy exported by the given runs: rollout under evaluation conditions, then a close-up
# contact sheet and a text summary in <run>/hands/. Exits once every run has written ALL DONE and is reported.
# usage: watch_hands.sh <gpu> <run_dir> [<run_dir> ...]
set -u
GPU=$1
shift
ROOT=$HOME/track2
PY=$HOME/video_to_data/robotic_grounding/flash_chord/.venv/bin/python
RENDER=$ROOT/.render_venv/bin/python
while true; do
  PENDING=0
  for D in "$@"; do
    OUT=$D/hands
    mkdir -p "$OUT"
    for CKPT in $(ls -tr "$D"/checkpoints/policy_*.safetensors 2>/dev/null | grep -v sonic_only); do
      S=$(basename "$CKPT" .safetensors)
      [ -f "$OUT/$S.done" ] && continue
      PENDING=1
      [ $(( $(date +%s) - $(stat -c %Y "$CKPT") )) -lt 30 ] && continue  # still being written
      NPZ=$OUT/$S.npz
      cd "$ROOT" || exit 1
      if CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$ROOT XLA_PYTHON_CLIENT_PREALLOCATE=false \
           "$PY" -m supermimic.record_rollout "$CKPT" "$NPZ" > "$OUT/$S.log" 2>&1 \
         && CUDA_VISIBLE_DEVICES=$GPU PYOPENGL_PLATFORM=egl \
           "$RENDER" "$ROOT/supermimic/render_hands.py" "$NPZ" "$OUT/$S" "$(basename "$D") $S" >> "$OUT/$S.log" 2>&1; then
        date '+%F %T' > "$OUT/$S.done"
      else
        date '+%F %T failed' > "$OUT/$S.done"
        touch "$OUT/$S.failed"
      fi
      rm -f "$NPZ"
    done
    grep -q "ALL DONE" "$D.status.log" 2>/dev/null || PENDING=1
  done
  [ "$PENDING" -eq 0 ] && break
  sleep 60
done
