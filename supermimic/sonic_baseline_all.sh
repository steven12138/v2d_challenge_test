#!/usr/bin/env bash
# Official evaluation of pure SONIC per episode; lines of "episode<TAB>sequence" on stdin. Results in logs/sonic_baseline/.
# usage: sonic_baseline_all.sh <gpu> < list
GPU=$1
ROOT=$HOME/track2
PY=$HOME/video_to_data/robotic_grounding/flash_chord/.venv/bin/python
OUT=$ROOT/logs/sonic_baseline
cd "$ROOT" || exit 1
while IFS=$'\t' read -r EP SEQ; do
  D=$OUT/ep$(printf '%02d' "$EP")
  [ -f "$D/done" ] && continue
  mkdir -p "$D"
  CK=$D/policy_sonic_only.safetensors
  if CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$ROOT "$PY" -u -m supermimic.sonic_baseline \
       "$ROOT/data/retargeted/t1/whole_body/soma/sequence_id=$SEQ/robot_name=g1/data.parquet" "$CK" > "$D/export.log" 2>&1 \
     && CUDA_VISIBLE_DEVICES=$GPU bash "$ROOT/supermimic/eval_episode.sh" "$CK" "$D/eval" "$EP" > "$D/eval.log" 2>&1; then
    date '+%F %T' > "$D/done"
  fi
  echo "ep$EP $SEQ $(grep -h 'locally computed' "$D/eval.log" | tail -1 | sed 's/.*computed: //') | $(grep -h SEGMENTS "$D/eval.log" | tail -1)" >> "$OUT/summary.txt"
done
