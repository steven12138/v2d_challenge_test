#!/bin/bash
# Copy the code to Grace or FASTER (they share /scratch): scripts/sync_to_grace.sh [grace|faster]
# Needs an open ssh session to that host. external/video_to_data is reference only
# (and external/holosoma) stay local; the official submission kit is synced because evaluation uses it.
set -euo pipefail
cd "$(dirname "$0")/.."

rsync -a --delete \
    --exclude .git --exclude __pycache__ --exclude external/video_to_data --exclude external/holosoma --exclude .venv --exclude work/ --exclude videos/ \
    --exclude 'retarget/robotic_grounding/robotic_grounding/assets/body_models/soma/*' \
    ./ "${1:-grace}":/scratch/user/the/v2d/t2/code/
