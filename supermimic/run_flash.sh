#!/usr/bin/env bash
# Official FlashSAC recipe with the object keypoint reward of flash_objective.py (PLAN.md 13.22.1), ~2.3 h per episode on one GPU.
# Its checkpoints name supermimic's objective: the watchers import it through PYTHONPATH, the official evaluation uses the
# rewritten copy in <run>/official/. <run>_w is the view watch_stats.sh and watch_hands.sh read (<run>_w/checkpoints).
# usage: run_flash.sh <episode> <sequence_id> <gpu> [hydra overrides...]
# env: TAG=<suffix>, OBJECTIVE=official|keypoints, SEED=<seed, default 42>, NO_FINAL_EVAL=1 skips the official repeats of the final policy,
#      MOTION=<retargeted motion root, default data/retargeted/t1>
set -u
EP=$1 SEQ=$2 GPU=$3
shift 3
TARGET=supermimic.flash_objective.KeypointObjectiveConfig
[ "${OBJECTIVE:-keypoints}" = official ] && TARGET=flash_chord.objectives.recon_body.ReconBodyObjectiveConfig
MOTION=${MOTION:-$HOME/track2/data/retargeted/t1}
PQ=$MOTION/whole_body/soma/sequence_id=$SEQ/robot_name=g1/data.parquet
D=$HOME/track2/logs/flash_ep${EP}${TAG:-}
LOG=${D}_w.status.log
PY=$HOME/video_to_data/robotic_grounding/flash_chord/.venv/bin/python
SUPPORT=()
grep -qE "def (Cube|Cylinder)" "$MOTION/whole_body/reconstructed_stage/${SEQ}_support.usda" || SUPPORT=(scene.support=false)
mkdir -p "$D" "${D}_w" "$D/official"
ln -sfn "$D" "${D}_w/checkpoints"
echo "$(date '+%F %T') ep$EP gpu$GPU start" >> "$LOG"
cd ~/video_to_data/robotic_grounding/flash_chord || exit 1
CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$HOME/track2 "$PY" -u -m supermimic.train_guard \
  experiment=g1_recon_body_flash_sac "task.parquet='$PQ'" reset.seed="${SEED:-42}" logging.mode=disabled output_dir="$D" \
  objective._target_=$TARGET "${SUPPORT[@]}" \
  "$@" > "$D/stdout.log" 2>&1
rc=$?
echo "$(date '+%F %T') ep$EP exit=$rc" >> "$LOG"
if [ "$rc" -ne 0 ]; then
  exit "$rc"
fi
if [ -n "${NO_FINAL_EVAL:-}" ]; then
  echo "$(date '+%F %T') ep$EP ALL DONE" >> "$LOG"
  exit 0
fi
FINAL=$(ls -v "$D"/policy_*.safetensors | tail -1)
OFFICIAL=$D/official/$(basename "$FINAL")
PYTHONPATH=$HOME/track2 "$PY" -m supermimic.kp_checkpoint "$FINAL" "$OFFICIAL" >> "$LOG" 2>&1
CUDA_VISIBLE_DEVICES=$GPU bash ~/track2/supermimic/eval_episode.sh "$OFFICIAL" "$D/eval_repeat_final" "$EP" 3 > "$D/eval_repeat_final.log" 2>&1
echo "$(date '+%F %T') ep$EP repeated eval of $(basename "$FINAL") done" >> "$LOG"
echo "$(date '+%F %T') ep$EP ALL DONE" >> "$LOG"
