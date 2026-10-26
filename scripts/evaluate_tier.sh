#!/bin/bash
# Evaluate every trained policy of a tier with the official kit, one Grace A100 job per episode in parallel.
#   scripts/evaluate_tier.sh <dataset_dir> <train_dir> <results_dir>
# Writes <results_dir>/episode_NNNNNN.{parquet,json} (the packer's input) and <results_dir>_checkpoints/*.json (outside <results_dir>: the packer reads every JSON in it).
set -euo pipefail
cd /scratch/user/the/v2d/t2/code
source slurm/env.sh rg
mkdir -p "$3_checkpoints"
python evaluate/checkpoints.py "$1" "$2" "$3_checkpoints/all.json"
python - "$3_checkpoints" <<'PY'
import json, sys
from pathlib import Path
folder = Path(sys.argv[1])
for episode, checkpoint in json.loads((folder / "all.json").read_text()).items():
    (folder / f"episode_{int(episode):06d}.json").write_text(json.dumps({episode: checkpoint}) + "\n")
PY
for map in "$3"_checkpoints/episode_*.json; do
    name=$(basename "$map" .json)
    [[ -f $3/$name.parquet ]] && { echo "skip $name (evaluated)"; continue; }
    echo "$name: $(sbatch --parsable --job-name="ev-${name#episode_}" slurm/evaluate.sbatch "$map" "$3")"
done
