#!/usr/bin/env bash
# SuperMimic refinement that keeps the kp + average policy (PLAN.md 13.22.3): only the official and keypoint rewards, official termination,
# 3x lower actor lr, a trust region around the start policy, and an EMA evaluated in the simulator every 100 iterations.
# usage: run_anchor.sh <episode> <sequence_id> <gpu> <policy.safetensors> [extra overrides...]   env: TAG=<suffix>
set -u
EP=$1 SEQ=$2 GPU=$3 POLICY=$4
shift 4
export NO_FINAL_EVAL=1
exec bash "$HOME/track2/supermimic/run_grasp2.sh" "$EP" "$SEQ" "$GPU" \
  resume_policy="$POLICY" start_frac=0.7 stop_frac=0.85 resume_voc_level=0 export_every_iters=100 export_eval=false \
  actor_lr=1.5e-5 w_anchor=20 w_object_keypoints=3 w_object_keypoints_fine=3 sat_penalty=0 pre_lift_boost=1 \
  'frame0_reset_schedule=[[0.0,0.5]]' 'frame0_seed_schedule=[[0.0,0.5]]' 'termination_overrides={}' \
  w_object_orientation=0 w_object_orientation_coarse=0 w_object_position_fine=0 w_object_position_coarse=0 w_hand_in_object=0 \
  w_grasp_contact=0 w_object_velocity=0 w_pelvis_angular_stability=0 w_torso_sideways_lean=0 w_pelvis_velocity=0 \
  w_pelvis_angular_velocity=0 w_body_joint_velocity=0 w_finger_grip=0 w_residual_legs=0 w_residual_arms=0 \
  "$@"
