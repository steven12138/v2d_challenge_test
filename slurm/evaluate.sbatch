#!/bin/bash
# sbatch slurm/evaluate.sbatch <checkpoints.json> <results_dir>
# Runs the official submission kit's policy evaluation (one world per episode, as on the leaderboard).
# scripts/evaluate_tier.sh submits one of these per episode so all episodes run in parallel.
#SBATCH --job-name=t2-evaluate
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

python external/v2d_submission_kit/examples/run_policy_evaluation.py \
    --checkpoints "$1" --results "$2" --runtime-root train/flash_chord
