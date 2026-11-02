#!/usr/bin/env bash
# 64-world evaluation statistics for every policy exported by the given runs (<run>/hands/<policy>.stats).
# usage: watch_stats.sh <gpu> <parquet-sequence> <run_dir> ...   pairs are given as run_dir:sequence_id
set -u
GPU=$1
shift
ROOT=$HOME/track2
PY=$HOME/video_to_data/robotic_grounding/flash_chord/.venv/bin/python
while true; do
  PENDING=0
  for PAIR in "$@"; do
    D=${PAIR%%:*} SEQ=${PAIR##*:}
    mkdir -p "$D/hands"
    for CKPT in $(ls -tr "$D"/checkpoints/policy_*.safetensors 2>/dev/null | grep -v sonic_only); do
      S=$(basename "$CKPT" .safetensors)
      [ -f "$D/hands/$S.stats" ] && continue
      PENDING=1
      [ $(( $(date +%s) - $(stat -c %Y "$CKPT") )) -lt 30 ] && continue
      cd "$ROOT" || exit 1
      CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$ROOT "$PY" -u -m supermimic.diag_eval_stats \
        "$ROOT/data/retargeted/t1/whole_body/soma/sequence_id=$SEQ/robot_name=g1/data.parquet" "$CKPT" 2>/dev/null \
        | grep -E "policy_" > "$D/hands/$S.stats.tmp"
      mv "$D/hands/$S.stats.tmp" "$D/hands/$S.stats"
    done
    grep -q "ALL DONE" "$D.status.log" 2>/dev/null || PENDING=1
  done
  [ "$PENDING" -eq 0 ] && break
  sleep 60
done
