#!/bin/bash
# Train every episode listed as sub_pink_kp in episodes.txt on one Grace A100 (~3.3 h). Already started outputs are skipped.
# usage: ACCOUNT=<slurm account> bash scripts/submit_grace.sh      run from the Grace code directory, which holds slurm/ and train/
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
W=${WORK:-/scratch/user/the/v2d/t2/work/tier1}
O=$W/assets/human_motion_data
while read -r e seq src; do
  [ "$src" = sub_pink_kp ] || continue
  P=$O/whole_body/soma/sequence_id=$seq/robot_name=g1/data.parquet
  [ -f "$P" ] || { echo "ep$e missing motion"; continue; }
  [ -e "$W/$src/$seq" ] && { echo "ep$e exists, skipped"; continue; }
  extra=()
  grep -qE "def (Cube|Cylinder)" "$O/whole_body/reconstructed_stage/${seq}_support.usda" || extra=(scene.support=false)
  sbatch --parsable --account="$ACCOUNT" --time=6:00:00 --job-name="ka-ep$e" --export=ALL,EXPERIMENT=g1_recon_body_kp_flash_sac \
    slurm/train_a100.sbatch "$P" "$W/$src/$seq" "${extra[@]}"
done < "$HERE/episodes.txt"
