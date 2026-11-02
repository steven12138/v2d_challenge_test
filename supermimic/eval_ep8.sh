#!/usr/bin/env bash
# Official evaluation of one ep-8 checkpoint: kit evaluator (FlashCHORD) -> kit packer (local metrics).
# usage: eval_ep8.sh <policy.safetensors> <output_dir> [repeats=1]
# The evaluator is not bitwise deterministic (MuJoCo-Warp contacts), so repeats>1 scores several rollouts.
set -euo pipefail
CKPT=$(realpath "$1")
OUT=$(realpath -m "$2")
REPEATS=${3:-1}
KIT=$HOME/track2/v2d_submission_kit
FC=$HOME/video_to_data/robotic_grounding/flash_chord
ROOT=$HOME/track2/kit_root
mkdir -p "$OUT"
echo "{\"8\": \"$CKPT\"}" > "$OUT/checkpoints.json"
cd "$KIT"
for i in $(seq 1 "$REPEATS"); do
  R="$OUT/results"
  [ "$REPEATS" -gt 1 ] && R="$OUT/results_$i"
  mkdir -p "$R"
  XLA_PYTHON_CLIENT_PREALLOCATE=false "$FC/.venv/bin/python" examples/run_policy_evaluation.py \
    --checkpoints "$OUT/checkpoints.json" --results "$R" --runtime-root "$FC"
  "$HOME/.local/bin/uv" run --no-project --with numpy --with pandas --with pyarrow --with trimesh \
    python tools/pack_submission.py --track track_2_tier1 --results "$R" \
    --dataset-root "$ROOT" --config "$HOME/track2/supermimic/tracks_ep8.json" \
    --code-commit https://github.com/supermimic/track2/commit/0000000 -o "$R/ep8.csv"
  "$FC/.venv/bin/python" "$HOME/track2/supermimic/analyze_eval.py" "$R/episode_000008.parquet" || true
done
