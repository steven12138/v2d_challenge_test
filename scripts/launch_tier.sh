#!/bin/bash
# Retarget (OmniRetarget) and train every sequence of one tier on Grace, all in parallel: one CPU retarget job per
# sequence, then one A100 training job per sequence that starts when its retarget succeeds.
#   scripts/launch_tier.sh <work_dir> <motion_root> <train_dir> [EXPERIMENT, default g1_recon_body_kp_flash_sac]
# <work_dir>/soma comes from prep/prep.py, and so does <motion_root>/whole_body/reconstructed_stage (its third
# argument). Sequences whose motion parquet exists skip retargeting; sequences with a final policy, a run
# writing to <train_dir>/<seq> or a queued training job are skipped, so the script can be re-run to resubmit failures. Extra sbatch options for the training jobs (e.g. FASTER's
# --partition=gpu --gres=gpu:a100:1) can be passed in TRAIN_SBATCH_OPTS.
set -euo pipefail
cd /scratch/user/the/v2d/t2/code
WORK=$1
MOTION=$2
TRAIN=$3
EXPERIMENT=${4:-g1_recon_body_kp_flash_sac}
read -r -a train_opts <<< "${TRAIN_SBATCH_OPTS:-}"

for soma in "$WORK"/soma/sequence_id=*; do
    seq=${soma##*/sequence_id=}
    parquet=$MOTION/whole_body/soma/sequence_id=$seq/robot_name=g1/data.parquet
    out=$TRAIN/$seq
    if [[ -f $out/policy_349999104.safetensors || -f $out/metrics.jsonl ]]; then
        echo "skip $seq (trained or training)"; continue
    fi
    # A queued training job is left alone, unless its retarget failed (dependency never satisfied).
    queued=$(squeue -h -u "$USER" -n "tr-${seq:20:20}" -o "%i %r")
    if [[ -n $queued ]]; then
        if [[ $queued == *DependencyNeverSatisfied* ]]; then
            scancel "${queued%% *}"
        else
            echo "skip $seq (queued: $queued)"; continue
        fi
    fi
    dependency=()
    if [[ ! -f $parquet ]]; then
        retarget=$(sbatch --parsable --job-name="om-${seq:20:20}" slurm/omniretarget.sbatch "$soma" "$WORK/scene" "$MOTION")
        dependency=(--dependency=afterok:"$retarget")
    fi
    # Ground-only sequences ship an empty support stage, which FlashCHORD rejects: turn support off.
    extra=()
    grep -qE "def (Cube|Cylinder)" "$MOTION/whole_body/reconstructed_stage/${seq}_support.usda" || extra=(scene.support=false)
    train=$(sbatch --parsable "${dependency[@]}" "${train_opts[@]}" --job-name="tr-${seq:20:20}" \
        --export=ALL,EXPERIMENT="$EXPERIMENT" slurm/train_a100.sbatch "$parquet" "$out" "${extra[@]}")
    echo "$seq: ${dependency[*]:-motion exists} -> train $train ${extra[*]}"
done
