#!/usr/bin/env bash
# For every finished clip of this version: render one evaluator-condition rollout (1 world, not the best of many), add the policy to
# checkpoints/, and refresh the table of official results. Videos land in logs/videos_v1/, one per episode.
# usage: watch_v1.sh <gpu>
set -u
GPU=$1
ROOT=$HOME/track2
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY=$HOME/video_to_data/robotic_grounding/flash_chord/.venv/bin/python
R=$ROOT/.render_venv/bin/python
OUT=$ROOT/logs/videos_v1
mkdir -p "$OUT"
while true; do
  for P in "$ROOT"/logs/kp_ep*/swa_last3.safetensors; do
    D=$(basename "$(dirname "$P")")
    E=${D#kp_ep}
    case "$E" in *[!0-9]*) continue ;; esac
    NAME=$(printf 'ep%02d' "$E")_$(awk -v e="ep$E" '$1 == e {print substr($2, 21)}' "$ROOT/logs/sonic_baseline/summary.txt" | head -1)
    [ -f "$OUT/$NAME.mp4" ] && continue
    (cd "$ROOT" && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$ROOT XLA_PYTHON_CLIENT_PREALLOCATE=false "$PY" -m supermimic.record_rollout "$P" "$OUT/$NAME.npz" 1) > "$OUT/$NAME.log" 2>&1 || continue
    CUDA_VISIBLE_DEVICES=$GPU PYOPENGL_PLATFORM=egl "$R" "$ROOT/supermimic/render_rollout.py" "$OUT/$NAME.npz" "$OUT/$NAME.tmp.mp4" "$NAME: kp + average of last 3" >> "$OUT/$NAME.log" 2>&1 \
      && mv "$OUT/$NAME.tmp.mp4" "$OUT/$NAME.mp4"
    CUDA_VISIBLE_DEVICES=$GPU PYOPENGL_PLATFORM=egl "$R" "$ROOT/supermimic/render_hands.py" "$OUT/$NAME.npz" "$OUT/${NAME}_hands" "$NAME" >> "$OUT/$NAME.log" 2>&1
    rm -f "$OUT/$NAME.npz"
    bash "$HERE/scripts/freeze_add.sh" "$E" "$P" swa_last3
  done
  python3 "$HERE/scripts/v1_table.py" "$ROOT/logs" > "$OUT/results.txt" 2>&1
  sleep 120
done
