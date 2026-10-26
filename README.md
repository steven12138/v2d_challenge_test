# V2D Challenge Track 2: robotic grounding baseline

This is the official Track 2 baseline for Tier 1 and Tier 2, run without the official Docker tooling.
The pipeline is: retarget SOMA-X → G1 with Pink IK, train FlashCHORD `g1_recon_body_flash_sac` (one
policy per episode), then evaluate and pack with the official submission kit. Design notes are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Setup (once)

```bash
scripts/sync_to_grace.sh                                   # local, with an open `ssh grace` session
ssh grace 'bash /scratch/user/the/v2d/t2/code/scripts/setup_grace.sh'
```

## Run one tier on Grace

Run these from `/scratch/user/the/v2d/t2/code`. The example uses Tier 1; for Tier 2 swap the dataset
folder and the work dir.

```bash
DATA=/scratch/user/the/v2d/data/challenge/track_2/tier_1_multiview_caption
WORK=/scratch/user/the/v2d/t2/work/tier1
MOTION=$WORK/assets/human_motion_data   # official layout: mesh paths under /assets/ stay portable

# 1. prep (login node, a few seconds)
source slurm/env.sh rg
python prep/prep.py $DATA $WORK/soma $MOTION/whole_body/reconstructed_stage

# 2. retarget, one CPU job per sequence
for seq in $WORK/soma/sequence_id=*; do sbatch slurm/retarget.sbatch $seq $MOTION; done

# 3. train, one A40 job per sequence
for parquet in $MOTION/whole_body/soma/sequence_id=*/robot_name=g1/data.parquet; do
    seq=$(basename $(dirname $(dirname $parquet)) | sed 's/sequence_id=//')
    # Ground-only sequences ship an empty support stage, which FlashCHORD rejects: turn support off.
    grep -qE "def (Cube|Cylinder)" $MOTION/whole_body/reconstructed_stage/${seq}_support.usda || extra=scene.support=false
    sbatch slurm/train.sbatch $parquet $WORK/train/$seq ${extra:-}; unset extra
done

# 4. evaluate with the official kit (needs all 30 final policies)
source slurm/env.sh flash
python evaluate/checkpoints.py $DATA $WORK/train $WORK/checkpoints.json
sbatch slurm/evaluate.sbatch $WORK/checkpoints.json $WORK/results

# 5. pack and upload with the official kit (login node). Use --track track_2_tier2 for Tier 2.
cd external/v2d_submission_kit
python tools/pack_submission.py --track track_2_tier1 --results $WORK/results \
    --dataset-root /scratch/user/the/v2d/data/challenge --code-commit "$COMMIT_URL" -o $WORK/track_2_tier1.csv
python submit/v2d_submit.py $WORK/track_2_tier1.csv --track track_2_tier1 -m "baseline" --dry-run
```

Render a motion locally (needs `.venv`, see `visualize/render.sh`). Add `scene.support=false` for
ground-only sequences:

```bash
visualize/render.sh <.../robot_name=g1/data.parquet> videos/<seq> <frames> [scene.support=false]
# the SOMA-X reconstruction (input of retargeting) in the same world and camera, for comparison:
.venv/bin/python visualize/render_input.py <soma_dir>/sequence_id=<seq> videos/<seq>/input.mp4
```

On a 40 GB A100 (partition `gpu` on Grace and FASTER) use `slurm/train_a100.sbatch` with the same
arguments: it halves the replay capacity to 5M to fit. On FASTER, train.sbatch needs
`--partition=gpu --gres=gpu:a40:1` on the sbatch command line.

A training job stopped by the 4-day limit is resumed from its newest state checkpoint:

```bash
sbatch slurm/train.sbatch $parquet $WORK/train/$seq \
    training.resume=true training.checkpoint_path=$WORK/train/$seq/state_<steps>.safetensors
```
