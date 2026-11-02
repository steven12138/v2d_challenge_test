#!/usr/bin/env bash
# ep 7 single-factor comparison of the two suspects for the worse body tracking (PLAN.md 13.20.4). Both keep the grip reward.
# A: old termination rule and frame-0 share, SONIC gradient off.  B: new termination rule, SONIC gradient on.
# usage: run_ep7_ab.sh A|B|<tag> <gpu> [extra overrides...]  (anything but A is variant B)
V=$1 GPU=$2
shift 2
SEQ=2026-05-29_15-20-43_iron_round_table_02
ARGS=(train_iters=10000 start_frac=0.2 export_every_iters=500 w_finger_grip=5.0 finger_grip_frac=0.6
      resume_policy=$(ls $HOME/track2/logs/batch_20261030/ep07/checkpoints/policy_final_*.safetensors | head -1))
if [ "$V" = "A" ]; then
  ARGS+=('policy_termination={}'
         'termination_overrides={"palm_position_threshold": 0.25, "object_position_threshold": 0.15, "object_orientation_threshold": 2.5}'
         'frame0_reset_schedule=[[0.0,0.15],[0.6,0.3],[0.85,0.5]]' 'frame0_seed_schedule=[[0.0,0.2],[0.6,0.3],[0.85,0.5]]')
else
  ARGS+=(sonic_stop_grad=false)
fi
TAG=_$V bash ~/track2/supermimic/run_grasp.sh 7 $SEQ $GPU "${ARGS[@]}" "$@"
