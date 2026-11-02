#!/usr/bin/env bash
# Pick-up-from-the-floor episodes, one GPU, 12000-iteration schedule (~4.5 h on an unshared H100; PLAN.md 13.16).
# Stage A: VOC with gravity compensation and a stiff spring (400 N/m, 40 N s/m) carries the object along the
#          reference so the policy learns the whole episode; the VOC level decays only after check windows in
#          which frame-0 rollouts survive, carry the object and touch it (schedule = upper bound); stops at 0.25.
# Stage B: resume A's full state; VOC without gravity compensation / orientation PD keeps decaying to 0;
#          object threshold tightened to 0.20 m; contact-loss termination (10 steps) on reference-grasp frames.
# usage: run_lift.sh <episode> <sequence_id> <gpu> [extra overrides for both stages...]
# env:   RESUME_A=<state_*.pkl> continues stage A from a saved state; A_TAG=<suffix> names a fresh stage-A dir
set -u
EP=$1 SEQ=$2 GPU=$3
shift 3
cd ~/track2
PY=~/video_to_data/robotic_grounding/flash_chord/.venv/bin/python
PQ=$HOME/track2/data/retargeted/t1/whole_body/soma/sequence_id=$SEQ/robot_name=g1/data.parquet
A=$HOME/track2/logs/lift_ep${EP}_A${A_TAG:-}
B=$HOME/track2/logs/lift_ep${EP}
LOG=$B.status.log
COMMON=(train_iters=12000 export_every_iters=1000 sat_penalty=0.5 wm_updates_per_iter=8 voc_k_lin=400 voc_d_lin=40
        w_object_orientation=3.0 w_object_position_fine=3.0 w_hand_in_object=3.0 hio_std=0.12
        w_grasp_contact=3.0 w_object_velocity=2.0 w_object_position_coarse=2.0 obj_contact_gate=0.8
        voc_adaptive=true 'voc_schedule=[[0.0,1.0],[0.15,1.0],[0.75,0.0],[1.0,0.0]]' "$@")
mkdir -p $A $B
echo "$(date '+%F %T') ep$EP gpu$GPU stage A start" >> $LOG
CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$HOME/track2 $PY -u -m supermimic.main \
  parquet=$PQ episode_index=$EP out_dir=$A ${RESUME_A:+resume_state=$RESUME_A} stop_frac=0.6 voc_stage_stop=0.25 voc_gravity_comp=true \
  'termination_overrides={"palm_position_threshold": 0.30, "object_position_threshold": 0.30, "object_orientation_threshold": 2.5}' \
  "${COMMON[@]}" > $A/stdout.log 2>&1
echo "$(date '+%F %T') ep$EP stage A exit=$?" >> $LOG
[ -f $A/checkpoints/state_final.pkl ] || { echo "$(date '+%F %T') ep$EP stage A FAILED (no state)" >> $LOG; exit 1; }
echo "$(date '+%F %T') ep$EP stage B start" >> $LOG
CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$HOME/track2 $PY -u -m supermimic.main \
  parquet=$PQ episode_index=$EP out_dir=$B resume_state=$A/checkpoints/state_final.pkl \
  voc_gravity_comp=false voc_angular_scale=0.0 contact_loss_steps=10 \
  'termination_overrides={"palm_position_threshold": 0.30, "object_position_threshold": 0.20, "object_orientation_threshold": 2.5}' \
  "${COMMON[@]}" > $B/stdout.log 2>&1
echo "$(date '+%F %T') ep$EP stage B exit=$?" >> $LOG
FINAL=$(ls -t $B/checkpoints/policy_final_*.safetensors 2>/dev/null | head -1)
if [ -n "$FINAL" ]; then
  CUDA_VISIBLE_DEVICES=$GPU bash ~/track2/supermimic/eval_episode.sh $FINAL $B/eval_repeat_final $EP 3 > $B/eval_repeat_final.log 2>&1
  echo "$(date '+%F %T') ep$EP repeated eval of $(basename $FINAL) done" >> $LOG
fi
echo "$(date '+%F %T') ep$EP ALL DONE" >> $LOG
