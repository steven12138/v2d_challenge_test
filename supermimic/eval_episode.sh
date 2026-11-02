#!/usr/bin/env bash
# Official evaluation of one checkpoint on one Track-2 episode: kit evaluator -> kit packer.
# usage: eval_episode.sh <policy.safetensors> <output_dir> <episode_index> [repeats=1]
# env: TRACK=track_2_tier1 (default) or track_2_tier2, FRAME_REF=<policy input in the released world frame, when it is not the released tier input>
set -euo pipefail
TRACK=${TRACK:-track_2_tier1}
CKPT=$(realpath "$1")
OUT=$(realpath -m "$2")
EP=$3
REPEATS=${4:-1}
KIT=$HOME/track2/v2d_submission_kit
FC=$HOME/video_to_data/robotic_grounding/flash_chord
ROOT=$HOME/track2/kit_root
REF=()
[ -n "${FRAME_REF:-}" ] && REF=(--frame-references "$FRAME_REF")
mkdir -p "$OUT"
TRACKS="$OUT/tracks_ep$EP.json"
python3 -c "import json,sys; d=json.load(open('$KIT/config/tracks.json')); d['tracks']['$TRACK']['scored_episodes']=[$EP]; json.dump(d, open('$TRACKS','w'), indent=1)"
echo "{\"$EP\": \"$CKPT\"}" > "$OUT/checkpoints.json"
cd "$KIT"
for i in $(seq 1 "$REPEATS"); do
  R="$OUT/results"
  [ "$REPEATS" -gt 1 ] && R="$OUT/results_$i"
  mkdir -p "$R"
  XLA_PYTHON_CLIENT_PREALLOCATE=false "$FC/.venv/bin/python" examples/run_policy_evaluation.py \
    --checkpoints "$OUT/checkpoints.json" --results "$R" --runtime-root "$FC"
  "$HOME/.local/bin/uv" run --no-project --with numpy --with pandas --with pyarrow --with trimesh \
    python tools/pack_submission.py --track "$TRACK" --results "$R" \
    --dataset-root "$ROOT" --config "$TRACKS" "${REF[@]}" \
    --code-commit https://github.com/supermimic/track2/commit/0000000 -o "$R/ep$EP.csv"
  "$FC/.venv/bin/python" "$HOME/track2/supermimic/analyze_eval.py" "$R/episode_$(printf %06d $EP).parquet" "$EP" || true
done
