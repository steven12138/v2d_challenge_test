#!/usr/bin/env bash
# Tier 2 run trained on Grace, its last 3 policies already in logs/flash_ep<E>_t2p: official config, average, then the statistics and official repeats that run_t2_refine.sh waits for.
# usage: run_t2_post.sh <episode> <sequence_id> <gpu>
set -u
EP=$1 SEQ=$2 GPU=$3
ROOT=$HOME/track2
LOGS=$ROOT/logs
FLASH=$LOGS/flash_ep${EP}_t2p
PY=$HOME/video_to_data/robotic_grounding/flash_chord/.venv/bin/python
note() { echo "$(date '+%F %T') ep$EP gpu$GPU $*" >> "$LOGS/pipeline.log"; }
export PYTHONPATH=$ROOT MOTION=$ROOT/data/retargeted/t2b TRACK=track_2_tier2 FRAME_REF=$ROOT/data/raw/tier_2_anchored
mkdir -p "$FLASH/official"
note "T2 post of the Grace run start"
INPUTS=()
for P in $(ls -v "$FLASH"/policy_*.safetensors | tail -3); do
  "$PY" -m supermimic.kp_checkpoint "$P" "$FLASH/official/$(basename "$P")" || { note "FAILED conversion _t2p"; exit 1; }
  INPUTS+=("$FLASH/official/$(basename "$P")")
done
if ! CUDA_VISIBLE_DEVICES=$GPU "$PY" -m supermimic.swa_policy "$FLASH/official/swa_last3.safetensors" "${INPUTS[@]}" > /dev/null 2>&1; then
  note "FAILED average _t2p"
  exit 1
fi
bash "$ROOT/supermimic/run_refine.sh" "$EP" "$SEQ" "$GPU" "$FLASH/official/swa_last3.safetensors"
