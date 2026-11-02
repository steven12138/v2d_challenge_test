#!/usr/bin/env bash
# 64-world statistics of the newest EMA policy of a SuperMimic run, skipping ahead when exports come faster. Written to logs/pipeline.log.
# usage: watch_ema.sh <gpu> <episode> <run_dir>   stops when <run_dir>.status.log says ALL DONE or exit
set -u
GPU=$1 EP=$2 D=$3
ROOT=$HOME/track2
PY=$HOME/video_to_data/robotic_grounding/flash_chord/.venv/bin/python
mkdir -p "$D/hands"
while true; do
  CK=$(ls -v "$D"/checkpoints/policy_ema_[0-9]*.safetensors 2>/dev/null | tail -1)
  if [ -n "$CK" ]; then
    S=$(basename "$CK" .safetensors)
    if [ ! -f "$D/hands/$S.stats" ] && [ $(( $(date +%s) - $(stat -c %Y "$CK") )) -ge 10 ]; then
      cd "$ROOT" || exit 1
      CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$ROOT "$PY" -u -m supermimic.diag_eval_stats x "$CK" 2>/dev/null | grep -E "policy_" > "$D/hands/$S.stats"
      echo "$(date '+%F %T') ep$EP gpu$GPU SS $(tr '\n' ' ' < "$D/hands/$S.stats" | cut -c1-330)" >> "$ROOT/logs/pipeline.log"
    fi
  fi
  grep -qE "ALL DONE" "$D.status.log" 2>/dev/null && break
  sleep 20
done
