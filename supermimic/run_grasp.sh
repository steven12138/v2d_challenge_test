#!/usr/bin/env bash
# Grasp recipe aligned with the official FlashCHORD curriculum (PLAN.md 13.20), one GPU, ~3.2 h on an unshared H100.
# Official VOC (50 N/m spring + gravity compensation: a weightless object the hands must carry), official stage values
# with 40 % of the iterations at VOC 0, settle resets at return-weighted frame windows (fingers close on the object
# under full VOC), body residual off during the settle, tighter termination, no contact-loss termination.
# usage: run_grasp.sh <episode> <sequence_id> <gpu> [extra overrides...]
# env:   TAG=<suffix> names the run dir
set -u
EP=$1 SEQ=$2 GPU=$3
shift 3
cd ~/track2
PY=~/video_to_data/robotic_grounding/flash_chord/.venv/bin/python
PQ=$HOME/track2/data/retargeted/t1/whole_body/soma/sequence_id=$SEQ/robot_name=g1/data.parquet
D=$HOME/track2/logs/grasp_ep${EP}${TAG:-}
LOG=$D.status.log
mkdir -p $D
echo "$(date '+%F %T') ep$EP gpu$GPU start" >> $LOG
CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$HOME/track2 $PY -u -m supermimic.main \
  parquet=$PQ episode_index=$EP out_dir=$D \
  world_count=8192 collect_steps_per_iter=2 imag_batch=6144 wm_batch=2048 wm_updates_per_iter=8 \
  train_iters=15000 export_every_iters=1500 lr_final_ratio=0.1 sat_penalty=0.5 \
  voc_gravity_comp=true 'voc_schedule=[[0.0,1.0],[0.1,1.0],[0.2,0.75],[0.3,0.5],[0.4,0.25],[0.5,0.1],[0.6,0.0],[1.0,0.0]]' \
  settle_reset_windows=true reset_bin_steps=25 pre_lift_boost=3.0 settle_policy_fingers_only=true \
  'frame0_reset_schedule=[[0.0,0.05],[0.6,0.1],[0.85,0.2]]' 'frame0_seed_schedule=[[0.0,0.2],[0.6,0.3]]' \
  wm_object_weight=5.0 sonic_stop_grad=true w_action_residual=5.0 policy_ema_decay=0.999 \
  w_object_orientation=3.0 w_object_position_fine=3.0 w_hand_in_object=3.0 hio_std=0.12 \
  w_grasp_contact=3.0 w_object_velocity=2.0 w_object_position_coarse=2.0 \
  w_pelvis_angular_stability=0.05 w_torso_sideways_lean=2.0 \
  w_pelvis_velocity=1.0 w_pelvis_angular_velocity=1.0 w_body_joint_velocity=1.0 \
  'termination_overrides={"pelvis_position_threshold": 0.5, "pelvis_orientation_threshold": 1.2, "palm_position_threshold": 99.0, "palm_orientation_threshold": 99.0, "object_position_threshold": 99.0, "object_orientation_threshold": 99.0}' \
  'policy_termination={"palm_position": 0.25, "palm_orientation": 1.5, "object_position": 0.15, "object_orientation": 2.5}' \
  "$@" > $D/stdout.log 2>&1
rc=$?
echo "$(date '+%F %T') ep$EP exit=$rc" >> $LOG
if [ "$rc" -ne 0 ]; then
  exit "$rc"
fi
FINAL=$(ls -t $D/checkpoints/policy_final_*.safetensors 2>/dev/null | head -1)
if [ -n "$FINAL" ]; then
  CUDA_VISIBLE_DEVICES=$GPU bash ~/track2/supermimic/eval_episode.sh $FINAL $D/eval_repeat_final $EP 3 > $D/eval_repeat_final.log 2>&1
  echo "$(date '+%F %T') ep$EP repeated eval of $(basename $FINAL) done" >> $LOG
fi
echo "$(date '+%F %T') ep$EP ALL DONE" >> $LOG
