#!/usr/bin/env bash
# Select the best checkpoint per episode, or take the map in env CHECKPOINTS, give each one fresh official run, and pack a Track 2 Tier 1 CSV. Uploads nothing.
# usage: [CHECKPOINTS=<map.json>] build_submission.sh <name> <commit url> [gpu ...]    writes ~/track2/submissions/<name>/
set -u
NAME=$1 URL=$2
shift 2
GPUS=("${@:-3}")
ROOT=$HOME/track2
KIT=$ROOT/v2d_submission_kit
FC=$HOME/video_to_data/robotic_grounding/flash_chord
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$ROOT/submissions/$NAME
mkdir -p "$OUT/results"
if [ -n "${CHECKPOINTS:-}" ]; then
  cp "$CHECKPOINTS" "$OUT/checkpoints.json"
else
  python3 "$HERE/scan_official.py" "$ROOT/logs" "$OUT/checkpoints.json" > "$OUT/selection.txt" 2>&1
fi
# one evaluator per GPU; each works through its own share of the episodes
python3 - "$OUT" "${#GPUS[@]}" <<'EOF'
import json, sys
out, n = sys.argv[1], int(sys.argv[2])
c = json.load(open(f"{out}/checkpoints.json"))
for i in range(n):
    json.dump({k: v for j, (k, v) in enumerate(sorted(c.items(), key=lambda x: int(x[0]))) if j % n == i}, open(f"{out}/share_{i}.json", "w"), indent=1)
EOF
cd "$KIT" || exit 1
PIDS=()
for I in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPUS[$I]} XLA_PYTHON_CLIENT_PREALLOCATE=false "$FC/.venv/bin/python" examples/run_policy_evaluation.py \
    --checkpoints "$OUT/share_$I.json" --results "$OUT/results" --runtime-root "$FC" > "$OUT/eval_$I.log" 2>&1 &
  PIDS+=($!)
done
RC=0
for P in "${PIDS[@]}"; do wait "$P" || RC=1; done
echo "evaluation rc=$RC; episodes: $(ls "$OUT"/results/episode_*.parquet 2>/dev/null | wc -l)" > "$OUT/pack.log"
"$HOME/.local/bin/uv" run --no-project --with numpy --with pandas --with pyarrow --with trimesh \
  python tools/pack_submission.py --track track_2_tier1 --results "$OUT/results" --dataset-root "$ROOT/kit_root" \
  --code-commit "$URL" -o "$OUT/$NAME.csv" >> "$OUT/pack.log" 2>&1
grep -E "locally computed|error|Error|rows|upload with" "$OUT/pack.log" | tail -n 6
echo "$(date '+%F %T') submission $NAME: $(grep -h 'locally computed' "$OUT/pack.log" | tail -1 | sed 's/.*computed: //')" >> "$ROOT/logs/pipeline.log"
