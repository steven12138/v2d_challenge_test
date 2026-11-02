#!/usr/bin/env bash
# Pack a Track 2 Tier 2 CSV, one fresh official run each: logs/t2_final/ep<E>.safetensors if refined, else the average of the last 3 policies of the given run tag. Uploads nothing.
# No checkpoint is chosen by its local score: that score reads the clean reference, so it is only reported (PLAN.md 13.22.8).
# usage: build_tier2.sh <name> <commit url> <run tag, e.g. _t2b> <gpu> [gpu ...]    writes ~/track2/submissions/<name>/; env FRAME_REF=<policy input>
set -u
NAME=$1 URL=$2 RTAG=$3
shift 3
GPUS=("$@")
ROOT=$HOME/track2
KIT=$ROOT/v2d_submission_kit
FC=$HOME/video_to_data/robotic_grounding/flash_chord
REF=${FRAME_REF:-$ROOT/data/raw/tier_2_anchored}
OUT=$ROOT/submissions/$NAME
mkdir -p "$OUT/results"
python3 - "$OUT" "${#GPUS[@]}" "$ROOT" "$RTAG" <<'EOF'
import json, os, sys
out, n, root, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
found = {}
for e in range(30):
    refined = f"{root}/logs/t2_final/ep{e}.safetensors"
    found[str(e)] = refined if os.path.exists(refined) else f"{root}/logs/flash_ep{e}{tag}/official/swa_last3.safetensors"
missing = [e for e, p in found.items() if not os.path.exists(p)]
print("missing episodes:", missing)
found = {e: p for e, p in found.items() if e not in missing}
json.dump(found, open(f"{out}/checkpoints.json", "w"), indent=1)
for i in range(n):
    json.dump({k: v for j, (k, v) in enumerate(sorted(found.items(), key=lambda x: int(x[0]))) if j % n == i}, open(f"{out}/share_{i}.json", "w"), indent=1)
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
  python tools/pack_submission.py --track track_2_tier2 --results "$OUT/results" --dataset-root "$ROOT/kit_root" --frame-references "$REF" \
  --code-commit "$URL" -o "$OUT/$NAME.csv" >> "$OUT/pack.log" 2>&1
grep -E "input reference check|locally computed|error|Error|rows|upload with" "$OUT/pack.log" | tail -n 6
