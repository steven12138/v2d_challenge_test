"""Hyperparameters. Paper values (SuperMimic Table A.5) unless noted as a FlashCHORD adaptation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # --- task ---
    parquet: str = (
        "/home/the/track2/data/retargeted/t1/whole_body/soma/"
        "sequence_id=2026-06-05_15-08-08_skinny_wood_chair_push_10/robot_name=g1/data.parquet"
    )
    episode_index: int = 8
    out_dir: str = "/home/the/track2/logs/supermimic_ep8"
    seed: int = 0

    # --- simulation / collection ---
    world_count: int = 8192
    buffer_steps: int = 1000  # per world (paper: 1000 steps per environment)
    action_noise: float = 0.7  # raw action units (tanh*10 space); FlashSAC target sigma 0.07 normalized
    collect_steps_per_iter: int = 1
    warmup_steps: int = 120
    frame0_reset_fraction: float = 0.3  # explicit frame-0 resets (= evaluation start state)
    return_reset_fraction: float = 0.35  # explicit resets at return-weighted bins; rest = official sampled reset
    reset_bin_steps: int = 50  # 1 s at 50 Hz
    # Official settle resets at return-weighted frame windows instead of explicit mid-episode resets (which place the
    # object in the air next to open hands). Each settle closes the fingers around it under full VOC.
    settle_reset_windows: bool = False
    settle_ok_palm: float = 0.3  # bins whose palm error at release exceeds this (fast frames: the frozen pose falls apart) are sampled 10x less
    pre_lift_boost: float = 1.0  # extra sampling weight for windows just before the reference lift
    release_boost: float = 1.0  # extra sampling weight for windows from shortly before the last reference hold to the end
    settle_policy_fingers_only: bool = False  # zero the body residual during the settle (the policy is not trained there)
    frame0_reset_schedule: tuple = ()  # piecewise-constant [[frac, value], ...] overriding frame0_reset_fraction
    frame0_seed_schedule: tuple = ()  # piecewise-constant [[frac, value], ...] overriding frame0_seed_fraction

    # --- world model ---
    wm_members: int = 3
    wm_hidden: int = 2048
    wm_layers: int = 3
    wm_window: int = 8  # k
    wm_batch: int = 4096  # windows per gradient step (per member)
    wm_updates_per_iter: int = 6
    wm_pretrain_updates: int = 3000
    wm_lr: float = 2.0e-4
    wm_object_weight: float = 1.0  # loss weight of the object dimensions of the state delta (15 of 140 dims)

    # --- policy / critic ---
    horizon_short: int = 16
    horizon_long: int = 32
    horizon_switch_frac: float = 0.2
    imag_batch: int = 12288  # divisible by wm_members
    imag_remat: bool = True  # rematerialize each imagined step in the backward pass (saves memory, costs a forward)
    sonic_stop_grad: bool = False  # no gradient through the SONIC decoder (its output is kept instead of recomputed)
    frame0_seed_fraction: float = 0.5
    gamma: float = 0.99
    lam: float = 0.95
    actor_lr: float = 5.0e-5
    critic_lr: float = 2.0e-4
    critic_hidden: int = 512
    critic_epochs: int = 5
    critic_minibatches: int = 4
    grad_clip_norm: float = 1.0
    grad_clip_value: float = 0.1
    lr_final_ratio: float = 0.03
    obs_var_floor: float = 1.0e-2
    critic_warmup_iters: int = 300  # critic-only updates before the actor moves
    term_tau_frac: float = 0.05  # soft-termination sigmoid width as a fraction of each threshold
    sat_threshold: float = 2.5  # |pre-tanh mean| above this is penalized (tanh(2.5)=0.987)
    sat_penalty: float = 0.0  # weight of the saturation penalty in the actor loss
    resume_state: str = ""  # path of a saved training state (state_*.pkl) to continue from
    resume_schedule_frac: float = -1.0  # >=0 restarts the schedule from this fraction while retaining saved weights/optimizers
    resume_voc_level: float = -1.0  # >=0 starts a resumed run at this assistance level instead of the saved level
    voc_gravity_comp: bool = True  # False: VOC keeps pose PD guidance but no gravity compensation
    voc_angular_scale: float = 1.0  # multiplier on the VOC orientation PD gains (training runtime only)
    voc_k_lin: float = 0.0  # >0 overrides the VOC position stiffness (training runtime only; official 50 N/m)
    voc_d_lin: float = 0.0  # >0 overrides the VOC position damping (official 10 N s/m)
    termination_overrides: dict = field(default_factory=dict)  # e.g. {"palm_position_threshold": 0.3}
    # Palm / object limits enforced by the collector and imagination instead of the env (keys of core termination: palm_position,
    # palm_orientation, object_position, object_orientation). Frame-0 episodes then only end on a fall, like the evaluator, so the
    # policy also learns to stay up after losing the object; every other episode ends at the limits as usual.
    policy_termination: dict = field(default_factory=dict)
    resume_policy: str = ""  # policy_*.safetensors to start the actor from (world model / critic retrained)
    start_frac: float = 0.0  # schedule position (VOC / lr / horizon) to start at when resuming a policy
    stop_frac: float = 1.0  # end this process at this schedule position (saves state_final.pkl to continue)

    # --- extra shaping (training only; exported config stays official) ---
    w_object_orientation: float = 1.0
    w_object_position_fine: float = 1.0
    w_object_keypoints: float = 0.0  # the Grace keypoint reward (std 0.1 m), which covers rotation like AUC and MPPE
    w_object_keypoints_fine: float = 0.0  # same with std 0.03 m, the accuracy where the AUC thresholds start
    w_anchor: float = 0.0  # cost on the action drift from the start policy (actions / 10, squared) on the seed states
    export_eval: bool = True  # official evaluation of every exported raw policy
    residual_hidden: int = 0  # >0: external residual branch of this width around the frozen resume_policy (residual.py)
    ff_knot_frames: int = 0  # >0: freeze resume_policy and train a feed-forward table of pre-tanh action offsets, one knot per this many frames
    ff_lr: float = 3.0e-3
    w_action_rate: float = 0.0  # !=0 replaces the official action rate weight (-1e-4); paper table A.2 is about -0.036 in these action units
    actor_lr_peak: float = 0.0  # >0: actor (or table) lr warms up to this after the critic warmup, then follows a cosine to actor_lr_end of it
    actor_lr_warmup: int = 300
    actor_lr_end: float = 0.05
    w_hand_in_object: float = 1.0
    hio_std: float = 0.05  # width of the hand-in-object-frame term (m)
    w_grasp_contact: float = 0.0  # hand links touching the object while the reference hand grasps it
    w_object_velocity: float = 0.0  # object linear velocity vs the (smoothed) reference velocity
    w_object_position_coarse: float = 0.0  # wide (0.4 m) object position term: gradient after a drop
    obj_contact_gate: float = 0.0  # on reference-grasp frames, object rewards *= 1 - gate*(1 - touch) (Omnigrasp)
    w_pelvis_angular_stability: float = 0.0  # cost on roll/pitch rate while approaching the object
    w_torso_sideways_lean: float = 0.0  # cost on lateral torso lean while approaching the object
    w_pelvis_velocity: float = 0.0  # track retargeted root translation speed
    w_pelvis_angular_velocity: float = 0.0  # track retargeted root angular speed
    w_body_joint_velocity: float = 0.0  # track retargeted G1 joint motion, not only joint poses
    w_object_orientation_coarse: float = 0.0  # wide (1 rad) orientation term: the 0.4 rad one has no gradient past about 1 rad
    w_residual_legs: float = 0.0  # cost on the leg and waist residual (a^2/100): falls grow with its size
    w_residual_arms: float = 0.0  # cost on the arm residual (a^2/100)
    w_action_residual: float = 0.0  # cost on the body residual (a^2/100, 29 joints): stay on SONIC unless it pays
    w_finger_grip: float = 0.0  # cost on the commanded finger pose vs a closed pose on holding frames (open otherwise)
    finger_grip_frac: float = 0.6  # closed pose: this fraction of the way to the farther joint limit
    grip_release_lead: int = 0  # open the fingers this many frames before each reference hold ends, before the palm leaves
    contact_loss_steps: int = 0  # >0: reset a world whose grasping hands touch nothing for this many steps
    fc_head_mix: float = 0.5  # FC reward = mix*learned head + (1-mix)*analytic soft FC

    # --- schedule ---
    train_hours: float = 8.0
    train_iters: int = 0  # >0: the schedule (VOC, lr, horizon, stop) advances per iteration, not per wall-clock second
    voc_schedule: tuple = ((0.0, 1.0), (0.10, 1.0), (0.60, 0.0), (1.0, 0.0))  # linear 1 -> 0 over 10%-60%
    # performance-triggered VOC decay (DexMachina): the level drops by `voc_decay_factor` after a check window in
    # which frame-0 rollouts survive, carry the object and touch it; `voc_schedule` stays an upper bound.
    voc_adaptive: bool = False
    voc_start: float = 1.0
    voc_decay_factor: float = 0.85
    voc_check_iters: int = 100
    voc_gate_len: float = 0.6  # mean frame-0 episode length / episode length
    voc_gate_carry: float = 0.6  # rows with the reference object lifted where the object is within 15 cm
    voc_gate_touch: float = 0.25  # link contact on reference-grasp frames (0..1)
    voc_stage_stop: float = 0.0  # >0: end this process once the VOC level is at or below this value
    voc_gate_after_lift: int = 0  # >0: length gate = mean frame-0 length >= first lifted reference frame + this
    voc_recover_patience: int = 0  # >0: raise the level one step after this many checks dying before the lift
    contact_loss_voc: float = 1.0  # contact-loss termination only while the VOC level is at or below this
    export_every_minutes: float = 15.0
    export_every_iters: int = 0  # >0: export / evaluate every this many iterations instead of minutes
    policy_ema_decay: float = 0.0  # >0: also export an exponential average of the actor as policy_ema_*.safetensors
    log_every_iters: int = 50
    eval_every_minutes: float = 30.0
    final_only: bool = False  # batch runs: skip baseline/intermediate exports and training-state snapshots

    extra_overrides: list = field(default_factory=list)
