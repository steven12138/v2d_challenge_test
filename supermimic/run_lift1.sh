#!/usr/bin/env bash
# Pick-up-from-the-floor episodes, single stage, one GPU, 12000-iteration schedule (PLAN.md 13.17).
# The whole VOC (gravity compensation, 400 N/m / 40 N s/m spring, orientation PD) scales with one level that only
# decays after check windows in which frame-0 rollouts last 50 frames past the reference lift, keep the object
# within 15 cm and touch it; three checks in a row dying before the lift raise it one step. The schedule is only a
# backstop (no forced decay before 40 %, zero at 95 %). Palm threshold 0.5 m; contact-loss termination once VOC <= 0.5.
# usage: run_lift1.sh <episode> <sequence_id> <gpu> [extra overrides...]
# env:   RESUME=<state_*.pkl> continues from a saved state; TAG=<suffix> names the run dir
set -u
EP=$1 SEQ=$2 GPU=$3
shift 3
cd ~/track2
PY=~/video_to_data/robotic_grounding/flash_chord/.venv/bin/python
PQ=$HOME/track2/data/retargeted/t1/whole_body/soma/sequence_id=$SEQ/robot_name=g1/data.parquet
D=$HOME/track2/logs/lift1_ep${EP}${TAG:-}
LOG=$D.status.log
mkdir -p $D
echo "$(date '+%F %T') ep$EP gpu$GPU single stage start ${RESUME:+(resume $RESUME)}" >> $LOG
CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$HOME/track2 $PY -u -m supermimic.main \
  parquet=$PQ episode_index=$EP out_dir=$D ${RESUME:+resume_state=$RESUME} \
  train_iters=12000 export_every_iters=1000 sat_penalty=0.5 wm_updates_per_iter=8 \
  voc_gravity_comp=true voc_k_lin=400 voc_d_lin=40 \
  w_object_orientation=3.0 w_object_position_fine=3.0 w_hand_in_object=3.0 hio_std=0.12 \
  w_grasp_contact=3.0 w_object_velocity=2.0 w_object_position_coarse=2.0 obj_contact_gate=0.8 \
  voc_adaptive=true 'voc_schedule=[[0.0,1.0],[0.4,1.0],[0.95,0.0],[1.0,0.0]]' \
  voc_gate_after_lift=50 voc_gate_carry=0.6 voc_gate_touch=0.25 voc_recover_patience=3 \
  contact_loss_steps=10 contact_loss_voc=0.5 \
  'termination_overrides={"palm_position_threshold": 0.50, "object_position_threshold": 0.30, "object_orientation_threshold": 2.5}' \
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
