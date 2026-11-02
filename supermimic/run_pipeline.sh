#!/usr/bin/env bash
# One episode with the frozen recipe on H100 (PLAN.md 13.22.2), ~2.3 h on one GPU: FlashSAC with the object keypoint reward, official
# config, average of the last 3 policies, then the statistics and official repeats of run_refine.sh.
# usage: run_pipeline.sh <episode> <sequence_id> <gpu>   env: SEED=<seed, default 2>; the run dir is flash_ep<E>_s<SEED>
set -u
EP=$1 SEQ=$2 GPU=$3 SEED=${SEED:-2}
ROOT=$HOME/track2
LOGS=$ROOT/logs
PY=$HOME/video_to_data/robotic_grounding/flash_chord/.venv/bin/python
note() { echo "$(date '+%F %T') ep$EP gpu$GPU $*" >> "$LOGS/pipeline.log"; }

note "kp training start seed=$SEED $SEQ"
if ! SEED=$SEED TAG=_s$SEED NO_FINAL_EVAL=1 OBJECTIVE=keypoints bash "$ROOT/supermimic/run_flash.sh" "$EP" "$SEQ" "$GPU"; then
  note "FAILED flashsac"
  exit 1
fi
FLASH=$LOGS/flash_ep${EP}_s$SEED
ls -v "$FLASH"/state_*.safetensors | head -n -1 | xargs -r rm -f
export PYTHONPATH=$ROOT
INPUTS=()
for P in $(ls -v "$FLASH"/policy_*.safetensors | tail -3); do
  "$PY" -m supermimic.kp_checkpoint "$P" "$FLASH/official/$(basename "$P")" || { note "FAILED conversion"; exit 1; }
  INPUTS+=("$FLASH/official/$(basename "$P")")
done
if ! CUDA_VISIBLE_DEVICES=$GPU "$PY" -m supermimic.swa_policy "$FLASH/official/swa_last3.safetensors" "${INPUTS[@]}" > /dev/null 2>&1; then
  note "FAILED average"
  exit 1
fi
bash "$ROOT/supermimic/run_refine.sh" "$EP" "$SEQ" "$GPU" "$FLASH/official/swa_last3.safetensors"
