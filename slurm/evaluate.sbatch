#!/bin/bash
# sbatch slurm/evaluate.sbatch <checkpoints.json> <results_dir>
# Runs the official submission kit's policy evaluation (one world per episode, as on the leaderboard).
#SBATCH --job-name=t2-evaluate
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/user/the/v2d/t2/logs/%x_%j.out

set -euo pipefail
cd /scratch/user/the/v2d/t2/code
source slurm/env.sh flash

python external/v2d_submission_kit/examples/run_policy_evaluation.py \
    --checkpoints "$1" --results "$2" --runtime-root train/flash_chord
