#!/usr/bin/env bash
# Feed-forward correction with SuperMimic (PLAN.md 13.22.6): the policy stays frozen and the world-model gradient trains a table of action
# offsets over time, which removes the repeatable object error. The EMA of each saved table is evaluated in the simulator (logs/pipeline.log).
# usage: run_ff.sh <tag> <episode> <sequence_id> <gpu> <policy.safetensors> [extra overrides...]
set -u
RTAG=$1 EP=$2 SEQ=$3 GPU=$4 POLICY=$5
shift 5
ROOT=$HOME/track2
PY=$HOME/video_to_data/robotic_grounding/flash_chord/.venv/bin/python
D=$ROOT/logs/grasp2_ep${EP}${RTAG}
mkdir -p "$D/hands"
(
  cd "$ROOT" || exit 1
  while true; do
    T=$(ls -v "$D"/checkpoints/ff_[0-9]*.npz 2>/dev/null | tail -1)
    if [ -n "$T" ]; then
      S=$(basename "$T" .npz)
      if [ ! -f "$D/hands/$S.stats" ] && [ $(( $(date +%s) - $(stat -c %Y "$T") )) -ge 5 ]; then
        CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$ROOT "$PY" -u -m supermimic.diag_eval_stats x "$POLICY::$T" 2>/dev/null | grep -E "ff_" > "$D/hands/$S.stats"
        echo "$(date '+%F %T') ep$EP gpu$GPU SS $(tr '\n' ' ' < "$D/hands/$S.stats" | cut -c1-330)" >> "$ROOT/logs/pipeline.log"
      fi
    fi
    grep -qE "ALL DONE" "$D.status.log" 2>/dev/null && break
    sleep 15
  done
) &
TAG=$RTAG bash "$ROOT/supermimic/run_anchor.sh" "$EP" "$SEQ" "$GPU" "$POLICY" ff_knot_frames=10 export_every_iters=200 stop_frac=1.0 \
  policy_ema_decay=0.995 w_anchor=0 "$@"
wait
