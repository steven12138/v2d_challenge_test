#!/usr/bin/env bash
# One FlashSAC recipe variant on H100, ~2.3 h on one GPU: keypoint reward plus the given overrides, statistics of the averaged policy from
# 290M steps on, then the average of the last 3 policies with its official repeats.
# usage: run_variant.sh <tag> <episode> <sequence_id> <gpu> [hydra overrides...]   the run dir is flash_ep<E><tag>
set -u
RTAG=$1 EP=$2 SEQ=$3 GPU=$4
shift 4
ROOT=$HOME/track2
LOGS=$ROOT/logs
PY=$HOME/video_to_data/robotic_grounding/flash_chord/.venv/bin/python
FLASH=$LOGS/flash_ep${EP}${RTAG}
note() { echo "$(date '+%F %T') ep$EP gpu$GPU $*" >> "$LOGS/pipeline.log"; }

mkdir -p "$FLASH"
note "kp variant $RTAG start $*"
nohup setsid bash "$ROOT/supermimic/watch_swa.sh" "$GPU" "$EP" "$FLASH" 290000000 > /dev/null 2>&1 < /dev/null &
if ! TAG=$RTAG NO_FINAL_EVAL=1 OBJECTIVE=keypoints bash "$ROOT/supermimic/run_flash.sh" "$EP" "$SEQ" "$GPU" "$@"; then
  note "FAILED variant $RTAG"
  exit 1
fi
ls -v "$FLASH"/state_*.safetensors | head -n -1 | xargs -r rm -f
export PYTHONPATH=$ROOT
INPUTS=()
for P in $(ls -v "$FLASH"/policy_*.safetensors | tail -3); do
  "$PY" -m supermimic.kp_checkpoint "$P" "$FLASH/official/$(basename "$P")" || { note "FAILED conversion $RTAG"; exit 1; }
  INPUTS+=("$FLASH/official/$(basename "$P")")
done
if ! CUDA_VISIBLE_DEVICES=$GPU "$PY" -m supermimic.swa_policy "$FLASH/official/swa_last3.safetensors" "${INPUTS[@]}" > /dev/null 2>&1; then
  note "FAILED average $RTAG"
  exit 1
fi
bash "$ROOT/supermimic/run_refine.sh" "$EP" "$SEQ" "$GPU" "$FLASH/official/swa_last3.safetensors"
