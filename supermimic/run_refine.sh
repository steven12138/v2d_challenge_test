#!/usr/bin/env bash
# SuperMimic refinement of a trained FlashSAC policy (PLAN.md 13.22), ~50 min on one H100: 64-world statistics and official repeats of
# the start policy, 1500 iterations from it (the EMA is the result), statistics of the EMAs, hand report, official repeats of the EMA.
# Progress in logs/pipeline.log.
# usage: run_refine.sh <episode> <sequence_id> <gpu> <policy.safetensors> [run tag, default _fw]
set -u
EP=$1 SEQ=$2 GPU=$3 POLICY=$4 RTAG=${5:-_fw}
ROOT=$HOME/track2
LOGS=$ROOT/logs
PY=$HOME/video_to_data/robotic_grounding/flash_chord/.venv/bin/python
RENDER=$ROOT/.render_venv/bin/python
note() { echo "$(date '+%F %T') ep$EP gpu$GPU $*" >> "$LOGS/pipeline.log"; }
stats() {  # stats <policy> <out>: 64 worlds under evaluation conditions
  (cd "$ROOT" && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$ROOT "$PY" -u -m supermimic.diag_eval_stats x "$1" 2>/dev/null) | grep -E "policy_|swa_" > "$2"
  note "SS $(tr '\n' ' ' < "$2" | cut -c1-330)"
}
official() {  # official <run log>: per-run official metrics
  note "EV $1: $(grep -h 'locally computed' "$1" | sed 's/locally computed: //; s/AUC=//; s/  SP-SR=/ SP /; s/  MP-SR=/ MP /; s/  MPPE=/ MPPE /' | tr '\n' ';')"
}

START=$(dirname "$POLICY")
mkdir -p "$START/hands"
NAME=$(basename "$POLICY" .safetensors)
stats "$POLICY" "$START/hands/$NAME.stats"
CUDA_VISIBLE_DEVICES=$GPU bash "$ROOT/supermimic/eval_episode.sh" "$POLICY" "$START/eval_repeat_$NAME" "$EP" 3 > "$START/eval_repeat_$NAME.log" 2>&1
official "$START/eval_repeat_$NAME.log"
if [ -f "$LOGS/pipeline.norefine" ]; then  # PLAN.md 13.22.2: the refinement damages kp + average policies; keep statistics and official repeats only
  note "DONE $RTAG no refine"
  exit 0
fi

note "refine start $POLICY"
if ! NO_FINAL_EVAL=1 TAG=$RTAG bash "$ROOT/supermimic/run_grasp2.sh" "$EP" "$SEQ" "$GPU" \
    resume_policy="$POLICY" start_frac=0.7 stop_frac=0.85 resume_voc_level=0 release_boost=6 grip_release_lead=25 \
    w_residual_legs=2 w_residual_arms=0 w_finger_grip=1 sat_penalty=0.1 \
    'frame0_reset_schedule=[[0.0,0.5]]' 'frame0_seed_schedule=[[0.0,0.5]]'; then
  note "FAILED refine"
  exit 1
fi
RUN=$LOGS/grasp2_ep${EP}${RTAG}
rm -f "$RUN"/checkpoints/state_[0-9]*.pkl
mkdir -p "$RUN/hands"
for CK in $(ls -v "$RUN"/checkpoints/policy_ema_[0-9]*.safetensors | tail -3); do
  stats "$CK" "$RUN/hands/$(basename "$CK" .safetensors).stats"
done
EMA=$(ls -v "$RUN"/checkpoints/policy_ema_final_*.safetensors | tail -1)
S=$(basename "$EMA" .safetensors)
if (cd "$ROOT" && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$ROOT XLA_PYTHON_CLIENT_PREALLOCATE=false "$PY" -m supermimic.record_rollout "$EMA" "$RUN/hands/$S.npz") > "$RUN/hands/$S.log" 2>&1 \
   && CUDA_VISIBLE_DEVICES=$GPU PYOPENGL_PLATFORM=egl "$RENDER" "$ROOT/supermimic/render_hands.py" "$RUN/hands/$S.npz" "$RUN/hands/$S" "ep$EP $S" >> "$RUN/hands/$S.log" 2>&1; then
  note "HD $(tail -n 4 "$RUN/hands/$S.txt" | head -1 | cut -c1-120)"
else
  note "hand report failed, see $RUN/hands/$S.log"
fi
rm -f "$RUN/hands/$S.npz"
CUDA_VISIBLE_DEVICES=$GPU bash "$ROOT/supermimic/eval_episode.sh" "$EMA" "$RUN/eval_repeat_ema" "$EP" 3 > "$RUN/eval_repeat_ema.log" 2>&1
official "$RUN/eval_repeat_ema.log"
note "DONE $RTAG"
