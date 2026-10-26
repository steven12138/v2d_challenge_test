#!/bin/bash
# sbatch slurm/retarget.sbatch <soma_dir>/sequence_id=<seq> <motion_root>
#SBATCH --job-name=t2-retarget
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/user/the/v2d/t2/logs/%x_%j.out

set -euo pipefail
cd /scratch/user/the/v2d/t2/code
source slurm/env.sh rg

python retarget/soma_to_g1.py "$1" --save --motion-root "$2"
