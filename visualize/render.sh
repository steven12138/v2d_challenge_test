#!/bin/bash
# Render a G1 motion parquet locally (needs .venv with FlashCHORD's runtime deps and a GPU).
#   visualize/render.sh <motion parquet> <out_dir> <frames> [extra Hydra overrides, e.g. scene.support=false]
# Writes <out_dir>/reference.mp4 (retargeted motion, kinematic, no physics) and <out_dir>/physical.mp4
# (official replay: SONIC with zero residual, physics on, i.e. the robot before any training).
set -euo pipefail
cd "$(dirname "$0")/.."

VIDEO=(viewer.backend=mp4 viewer.realtime=false viewer.video_fps=30 viewer.num_frames="$3"
       viewer.width=1280 viewer.height=720 markers.enabled=false start_paused=false
       "viewer.camera_position=[3.0,-3.0,1.8]" "viewer.camera_target=[0.0,0.0,0.7]"
       hydra.run.dir="$2/hydra")

.venv/bin/python visualize/render.py "task.parquet='$1'" viewer.output_path="$2/reference.mp4" "${VIDEO[@]}" "${@:4}"
.venv/bin/python train/flash_chord/scripts/debug/replay.py --config-name=sonic_replay \
    "task.parquet='$1'" viewer.output_path="$2/physical.mp4" "${VIDEO[@]}" "${@:4}"
