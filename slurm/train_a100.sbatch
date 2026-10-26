#!/bin/bash
# Same as train.sbatch, for one 40 GB A100 (partition "gpu" on both Grace and FASTER):
#   sbatch slurm/train_a100.sbatch <motion parquet> <out_dir> [extra Hydra overrides]
#
# The GPU is compute-bound (SM 100%, at its power limit), so one training per A100. It uses ~24 GB with a
# 5M replay. The official 10M replay ring adds ~11 GiB (~2.3 KB per transition), ~35 GB in total, so the
# recipe runs unchanged. If it runs out of memory, append training.replay.capacity=5000000.
# The trainer uses one CPU core and ~7 GB of host RAM. Grace bills max(cores, memory share), so asking
# for 4 cores and 32 GB halves the SU cost of the 8-core / 64 GB request. A run takes 3.5-4.5 h; the 10 h limit
# keeps the SUs Grace reserves up front (per hour of time limit) small.
#SBATCH --job-name=t2-train-a100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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
