#!/bin/bash
# sbatch slurm/train.sbatch <motion parquet> <out_dir> [extra Hydra overrides]
#
# To resume after the 4-day limit, pass the newest state checkpoint as extra overrides:
#   sbatch slurm/train.sbatch <parquet> <out_dir> training.resume=true training.checkpoint_path=<out_dir>/state_<steps>.safetensors
#SBATCH --job-name=t2-train
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/user/the/v2d/t2/logs/%x_%j.out

set -euo pipefail
cd /scratch/user/the/v2d/t2/code
source slurm/env.sh flash
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv

# Recipe: the official one unless EXPERIMENT is exported, e.g. sbatch --export=ALL,EXPERIMENT=g1_recon_body_kp_flash_sac
# SEED (default 42, the official one) is reset.seed; the official training.seed follows it (${reset.seed}).
python train/train.py experiment="${EXPERIMENT:-g1_recon_body_flash_sac}" \
    "task.parquet='$1'" \
    output_dir="$2" hydra.run.dir="$2/hydra" \
    reset.seed="${SEED:-42}" \
    "${@:3}"
