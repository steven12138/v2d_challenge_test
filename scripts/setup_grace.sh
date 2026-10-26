#!/bin/bash
# One-time setup on the Grace LOGIN node (compute nodes have no internet). Run after sync_to_grace.sh.
# The uv envs already hold the pinned dependencies; this only points them at the code in this repo.
set -euo pipefail
cd /scratch/user/the/v2d/t2/code
V2D=/scratch/user/the/v2d
UV=$V2D/bin/uv
mkdir -p $V2D/t2/logs

# rg: retarget/soma_to_g1.py imports retarget/robotic_grounding via sys.path; drop the stale install.
$UV pip uninstall --python $V2D/envs/rg/bin/python robotic_grounding || true

# flash: the official kit's evaluator imports flash_chord, so install train/flash_chord (editable).
$UV pip install --python $V2D/envs/flash/bin/python --no-deps -e train/flash_chord
$UV pip install --python $V2D/envs/flash/bin/python -r external/v2d_submission_kit/requirements.txt

# SOMA-X body-model assets -> retarget/robotic_grounding/robotic_grounding/assets/body_models/soma/
HF_HOME=$V2D/.hf_cache $V2D/envs/rg/bin/python retarget/setup_soma_assets.py --identity-model-type mhr
