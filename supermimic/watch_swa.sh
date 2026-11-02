#!/usr/bin/env bash
# 64-world statistics of the average of the last 3 policies, for every new checkpoint of a FlashSAC run. Written to logs/pipeline.log.
# usage: watch_swa.sh <gpu> <episode> <run_dir> [first step to evaluate]   stops when <run_dir>_w.status.log says ALL DONE or exit
set -u
GPU=$1 EP=$2 D=$3 FIRST=${4:-0}
ROOT=$HOME/track2
PY=$HOME/video_to_data/robotic_grounding/flash_chord/.venv/bin/python
mkdir -p "$D/swa"
while true; do
  mapfile -t P < <(ls -v "$D"/policy_*.safetensors 2>/dev/null)
  N=${#P[@]}
  if [ "$N" -ge 3 ]; then
    S=$(basename "${P[$((N - 1))]}" .safetensors)
    S=swa_${S#policy_}
    if [ "${S#swa_}" -ge "$FIRST" ] && [ ! -f "$D/swa/$S.stats" ] && [ $(( $(date +%s) - $(stat -c %Y "${P[$((N - 1))]}") )) -ge 30 ]; then
      cd "$ROOT" || exit 1
      export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$ROOT
      "$PY" -m supermimic.swa_policy "$D/swa/$S.safetensors" "${P[$((N - 3))]}" "${P[$((N - 2))]}" "${P[$((N - 1))]}" > /dev/null 2>&1
      "$PY" -u -m supermimic.diag_eval_stats x "$D/swa/$S.safetensors" 2>/dev/null | grep -E "swa_" > "$D/swa/$S.stats"
      echo "$(date '+%F %T') ep$EP gpu$GPU SS $(tr '\n' ' ' < "$D/swa/$S.stats" | cut -c1-330)" >> "$ROOT/logs/pipeline.log"
    fi
  fi
  grep -qE "ALL DONE|exit=" "${D}_w.status.log" 2>/dev/null && break
  sleep 30
done
