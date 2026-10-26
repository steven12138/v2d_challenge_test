#!/bin/bash
# Score one checkpoint with the official kit evaluator over many worlds, for stable method comparisons
# (a 1-world score is noisy: GPU contact solving is not bitwise deterministic).
#   sbatch slurm/eval_worlds.sbatch <policy_<steps>.safetensors> <out.json> [worlds, default 32]
#SBATCH --job-name=t2-evalN
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/user/the/v2d/t2/logs/%x_%j.out

set -euo pipefail
cd /scratch/user/the/v2d/t2/code
source slurm/env.sh flash
mkdir -p "$(dirname "$2")"
cd train/flash_chord

python ../../external/v2d_submission_kit/flash_chord/scripts/evaluate_policy.py \
    evaluation.checkpoint="$1" evaluation.world_count="${3:-32}" evaluation.reset_mode=explicit \
    evaluation.metrics_output="$2" hydra.run.dir="$(dirname "$2")/hydra_$(basename "$2" .json)" logging.mode=disabled
