#!/bin/bash
# Two Tier 1 batches from one Pink-IK retarget per sequence, all in parallel on Grace:
#   A  official Pink-IK motion                      -> object-keypoint recipe -> <work>/sub_pink_kp/<seq>
#   B  OmniRetarget refining the Pink-IK solution   -> object-keypoint recipe -> <work>/sub_refine_kp/<seq>
#   scripts/launch_tier1_ab.sh <work_dir>
# <work_dir> holds soma/ and assets/human_motion_data/whole_body/reconstructed_stage/ from prep/prep.py.
set -euo pipefail
cd /scratch/user/the/v2d/t2/code
WORK=$1
OFFICIAL=$WORK/assets/human_motion_data
REFINED=$WORK/refine/assets/human_motion_data
mkdir -p "$REFINED/whole_body/reconstructed_stage"
cp -n "$OFFICIAL"/whole_body/reconstructed_stage/*_support.usda "$REFINED/whole_body/reconstructed_stage/"

for soma in "$WORK"/soma/sequence_id=*; do
    seq=${soma##*/sequence_id=}
    short=${seq:20:16}
    official=$OFFICIAL/whole_body/soma/sequence_id=$seq/robot_name=g1/data.parquet
    refined=$REFINED/whole_body/soma/sequence_id=$seq/robot_name=g1/data.parquet
    extra=()
    grep -qE "def (Cube|Cylinder)" "$OFFICIAL/whole_body/reconstructed_stage/${seq}_support.usda" || extra=(scene.support=false)

    after_pink=()
    if [[ ! -f $official ]]; then
        pink=$(sbatch --parsable --job-name="pk-$short" slurm/retarget.sbatch "$soma" "$OFFICIAL")
        after_pink=(--dependency=afterok:"$pink")
    fi
    a=skip
    if [[ ! -f $WORK/sub_pink_kp/$seq/metrics.jsonl && ! -f $WORK/sub_pink_kp/$seq/policy_349999104.safetensors ]]; then
        a=$(sbatch --parsable "${after_pink[@]}" --job-name="ta-$short" --export=ALL,EXPERIMENT=g1_recon_body_kp_flash_sac \
            slurm/train_a100.sbatch "$official" "$WORK/sub_pink_kp/$seq" "${extra[@]}")
    fi
    omni=$(sbatch --parsable "${after_pink[@]}" --job-name="om-$short" \
        slurm/omniretarget.sbatch "$soma" "$WORK/scene" "$REFINED" "$official")
    b=$(sbatch --parsable --dependency=afterok:"$omni" --job-name="tb-$short" --export=ALL,EXPERIMENT=g1_recon_body_kp_flash_sac \
        slurm/train_a100.sbatch "$refined" "$WORK/sub_refine_kp/$seq" "${extra[@]}")
    echo "$short: pink ${after_pink[*]:-exists} | A $a | omni $omni -> B $b ${extra[*]}"
done
