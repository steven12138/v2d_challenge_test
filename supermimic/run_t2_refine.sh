#!/usr/bin/env bash
# Tier 2 worker: every finished FlashSAC run of the given tag gets two passes of SuperMimic refinement with the orientation terms (PLAN.md 13.22.8), each starting from the
# best policy so far by Tier 2 metrics. A refinement pass takes about 40 min next to a FlashSAC run. The chosen policy lands in logs/t2_final/ep<E>.safetensors.
# usage: run_t2_refine.sh <gpu> <run tag, e.g. _t2p> <motion root name>
set -u
GPU=$1 RTAG=$2 MOTION_NAME=$3
ROOT=$HOME/track2
LOGS=$ROOT/logs
PY=$HOME/video_to_data/robotic_grounding/flash_chord/.venv/bin/python
FINAL=$LOGS/t2_final
OPTS="world_count=4096 imag_batch=3072 w_anchor=5 actor_lr=3e-5 w_object_orientation=3 w_object_orientation_coarse=3"
export PYTHONPATH=$ROOT MOTION=$ROOT/data/retargeted/$MOTION_NAME
mkdir -p "$FINAL"
note() { echo "$(date '+%F %T') ep$1 gpu$GPU T2 $2" >> "$LOGS/pipeline.log"; }
sequence() {
  python3 - "$1" <<EOF
import json, sys
for line in open("$ROOT/data/raw/tier_2_synthetic_noise/meta/episodes_metadata.jsonl"):
    d = json.loads(line)
    if d["episode_index"] == int(sys.argv[1]):
        print(d["sequence_id"])
EOF
}
pick() {
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= "$PY" -m supermimic.t2_pick "$@" 2>> "$LOGS/t2_final/pick.log"
}
while true; do
  for BASE in "$LOGS"/flash_ep*"$RTAG"/official/swa_last3.safetensors; do
    [ -f "$BASE" ] || continue
    D=$(basename "$(dirname "$(dirname "$BASE")")")
    E=${D#flash_ep}
    E=${E%"$RTAG"}
    [ -e "$FINAL/ep$E.claim" ] && continue
    # the FlashSAC run writes its average before its official repeats: wait for them, they share the GPU
    [ "$(grep -c "locally computed" "$(dirname "$BASE")/eval_repeat_swa_last3.log" 2> /dev/null)" -ge 3 ] || continue
    (set -o noclobber; : > "$FINAL/ep$E.claim") 2> /dev/null || continue
    SEQ=$(sequence "$E")
    FIRST=$LOGS/grasp2_ep${E}_t2o
    SECOND=$LOGS/grasp2_ep${E}_t2o2
    note "$E" "refinement 1 start"
    bash "$ROOT/supermimic/test_anchor.sh" _t2o "$E" "$SEQ" "$GPU" "$BASE" $OPTS
    START=$(pick "$FIRST")
    if [ -z "$START" ]; then
      note "$E" "refinement 1 gave no policy"
      continue
    fi
    note "$E" "refinement 2 start from $(basename "$START")"
    bash "$ROOT/supermimic/test_anchor.sh" _t2o2 "$E" "$SEQ" "$GPU" "$START" $OPTS
    CHOSEN=$(pick "$FIRST" "$SECOND")
    cp "$CHOSEN" "$FINAL/ep$E.safetensors"
    note "$E" "chosen $(echo "$CHOSEN" | sed "s|$LOGS/||"): $(tail -n 1 "$LOGS/t2_final/pick.log")"
  done
  sleep 120
done
