#!/bin/bash
# Score one checkpoint with the official kit evaluator and record the policy (no VOC assistance).
#   sbatch slurm/review.sbatch <policy_<steps>.safetensors> <out_dir> <frames>
# Writes <out_dir>/metrics.json, <out_dir>/trajectory.parquet and <out_dir>/policy.mp4.
# Grace A100 by default; on FASTER add --partition=gpu --gres=gpu:a40:1 (or a100).
#SBATCH --job-name=t2-review
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/user/the/v2d/t2/logs/%x_%j.out

set -euo pipefail
cd /scratch/user/the/v2d/t2/code
source slurm/env.sh flash
export PYGLET_HEADLESS=true  # compute nodes have no display: pyglet renders through NVIDIA EGL
mkdir -p "$2"
cd train/flash_chord

python ../../external/v2d_submission_kit/flash_chord/scripts/evaluate_policy.py \
    evaluation.checkpoint="$1" evaluation.world_count=1 evaluation.reset_mode=explicit \
    evaluation.metrics_output="$2/metrics.json" evaluation.object_trajectories_output="$2/trajectory.parquet" \
    hydra.run.dir="$2/hydra_eval" logging.mode=disabled

python scripts/view_policy.py evaluation.checkpoint="$1" \
    viewer.backend=mp4 viewer.output_path="$2/policy.mp4" viewer.num_frames="$3" viewer.realtime=false \
    viewer.video_fps=30 viewer.width=1280 viewer.height=720 markers.enabled=false \
    "viewer.camera_position=[3.0,-3.0,1.8]" "viewer.camera_target=[0.0,0.0,0.7]" \
    hydra.run.dir="$2/hydra_view"
