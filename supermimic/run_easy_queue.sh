#!/usr/bin/env bash
# Simple hand episodes on one GPU, one after the other, continuing the policies of batch_20261030 (actor only: the
# world model has new inputs). The schedule starts at 20 % (VOC 0.75, horizon 32); 8000 iterations each.
# usage: run_easy_queue.sh <gpu> [extra overrides...]
GPU=$1
shift
B=$HOME/track2/logs/batch_20261030
run() {
  EP=$1 SEQ=$2
  shift 2
  bash ~/track2/supermimic/run_grasp.sh $EP $SEQ $GPU train_iters=10000 start_frac=0.2 export_every_iters=1000 \
    resume_policy=$(ls $B/ep$(printf '%02d' $EP)/checkpoints/policy_final_*.safetensors | head -1) "$@"
}
run 3 2026-04-27_11-50-10_contigo_coffee_mug_handle_table_05 "$@"
run 7 2026-05-29_15-20-43_iron_round_table_02 "$@"
run 6 2026-06-01_11-26-44_monitor_rotate_05 "$@"
