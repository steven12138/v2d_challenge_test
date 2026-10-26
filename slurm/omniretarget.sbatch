#!/bin/bash
# sbatch slurm/omniretarget.sbatch <soma_dir>/sequence_id=<seq> <scene_root> <motion_root>
# Builds the MuJoCo scene (<scene_root>/<seq>/) and retargets with OmniRetarget (CPU only, ~3-10 min per sequence).
#SBATCH --job-name=t2-omni
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/user/the/v2d/t2/logs/%x_%j.out

set -euo pipefail
cd /scratch/user/the/v2d/t2/code
source slurm/env.sh rg
seq=${1##*/sequence_id=}

python retarget/omniretarget/build_scene.py "$1/reconstructed_mesh/output_aligned.glb" "$2/$seq"
python retarget/omniretarget/omniretarget.py "$1" "$2/$seq" "$3"
