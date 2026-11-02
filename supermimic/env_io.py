"""Official FlashCHORD environment construction, constants, and zero-copy state readout."""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp

from flash_chord.configuration import compose_config, instantiate_typed, resolved_dict
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.objectives.wrench import friction_cone_angles, sample_wrench_basis
from flash_chord.runtime.jax import to_jax
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.training.checkpoint_metadata import (
    build_checkpoint_metadata,
    critic_schema,
    policy_schema,
    resolved_reference_config,
)
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.flash_sac.config import TrainingConfig

EXPERIMENT = "g1_recon_body_flash_sac"


@dataclass
class Handles:
    cfg: object
    env: RLEnv
    jax_env: WarpRLEnv
    training: TrainingConfig
    reference: object
    checkpoint_metadata: dict


def support_stage_is_empty(parquet: str) -> bool:
    """Ground-only episodes ship an empty support stage, which FlashCHORD's support loader rejects
    (scene/support.py raises on a stage without Cylinder/Cube prims). An empty stage adds no geometry,
    so building the scene with `scene.support=false` is physically identical."""
    from pathlib import Path

    p = Path(parquet)
    seq_dir = next(parent for parent in p.parents if parent.name.startswith("sequence_id="))
    seq = seq_dir.name.split("=", 1)[1]
    stage_dir = seq_dir.parent.parent / "reconstructed_stage"
    candidates = [stage_dir / f"{seq}_g1_support.usda", stage_dir / f"{seq}_support.usda"]
    stage = next((c for c in candidates if c.is_file()), None)
    if stage is None:
        return False
    text = stage.read_text()
    return ("Cube" not in text) and ("Cylinder" not in text)


def official_overrides(parquet: str) -> list[str]:
    overrides = [f"experiment={EXPERIMENT}", f"task.parquet='{parquet}'", "reset.seed=42", "logging.mode=disabled"]
    if support_stage_is_empty(parquet):
        overrides.append("scene.support=false")
    return overrides


def build(parquet: str, world_count: int, voc_gravity_comp: bool = True, voc_angular_scale: float = 1.0,
          termination_overrides: dict | None = None, voc_k_lin: float = 0.0, voc_d_lin: float = 0.0) -> Handles:
    """Build the official ReconBody env. Only the typed runtime copy differs from the official config:
    manual resets, and optionally VOC without gravity compensation (training-only; the checkpoint
    metadata below is always built from the untouched official config)."""
    cfg = compose_config("train_flash_sac", official_overrides(parquet))
    env_config = instantiate_typed(cfg.env, RLEnvConfig)
    env_config = replace(env_config, auto_reset=False)  # we log pre-reset terminal states, then reset
    if not voc_gravity_comp:
        # Gravity compensation makes the object weightless (and frictionless against the floor), which
        # teaches pushing strategies that fail without VOC. Keep only the pose PD guidance.
        env_config = replace(env_config, voc=replace(env_config.voc, gravity=0.0))
    if termination_overrides:
        # Training-only: the evaluator disables termination, and the score only looks at the object.
        env_config = replace(env_config, termination=replace(env_config.termination, **termination_overrides))
    if voc_angular_scale != 1.0:
        # Training-only: weaken/remove the VOC orientation PD so the policy itself must keep the object upright.
        env_config = replace(
            env_config,
            voc=replace(env_config.voc, k_ang=env_config.voc.k_ang * voc_angular_scale,
                        d_ang=env_config.voc.d_ang * voc_angular_scale),
        )
    if voc_k_lin > 0 or voc_d_lin > 0:
        # Training-only: the official 50 N/m, 10 N s/m spring lags a fast lift by 25-35 cm; a stiffer one
        # keeps the object near the reference while it is weightless, so the hands learn where it goes.
        env_config = replace(
            env_config,
            voc=replace(env_config.voc, k_lin=voc_k_lin if voc_k_lin > 0 else env_config.voc.k_lin,
                        d_lin=voc_d_lin if voc_d_lin > 0 else env_config.voc.d_lin),
        )
    embodiment = instantiate_typed(cfg.embodiment, Embodiment)
    collision = instantiate_typed(cfg.collision, CollisionPolicy)
    training = instantiate_typed(cfg.training, TrainingConfig)
    setup = setup_scene(
        parquet=cfg.task.parquet,
        control_fps=env_config.sim.fps,
        motion_speed=cfg.task.motion_speed,
        embodiment=embodiment,
        collision=collision,
        world_count=world_count,
        include_support=cfg.scene.support,
        decompose_objects=cfg.scene.decompose_objects,
        object_scale_min=cfg.scene.object_scale_min,
        object_scale_max=cfg.scene.object_scale_max,
        object_scale_seed=cfg.scene.object_scale_seed,
        contact_friction=cfg.scene.get("contact_friction", 1.0),
        object_free_joint_damping=cfg.scene.get("object_free_joint_damping", 0.0),
        source_frame_playback=bool(cfg.task.source_frame_playback),
        motion_start_frame=int(cfg.task.motion_start_frame),
        motion_end_frame=int(cfg.task.motion_end_frame),
    )
    env = RLEnv(setup.scene, setup.reference, config=env_config)
    env.reset()
    env.capture_step()
    env.reset()
    jax_env = WarpRLEnv(env, publish_diagnostics=False, publish_step_metrics=False)
    # Checkpoint metadata exactly as scripts/train_flash_sac.py builds it (official, unmodified config).
    environment_config = resolved_dict(cfg)
    environment_config["task"]["parquet"] = str(setup.parquet_path)
    environment_config["reference"] = resolved_reference_config(setup.reference, str(setup.parquet_path))
    metadata = build_checkpoint_metadata(
        environment_config,
        policy_schema(env.action, env.observation_strategy),
        critic_schema(env.action, env.observation_strategy, jax_env.critic_context_names),
    )
    return Handles(cfg, env, jax_env, training, setup.reference, metadata)


def _np(array) -> np.ndarray:
    return np.asarray(array.numpy())


@dataclass
class Consts:
    """Per-task constants consumed by the JAX observation, reward and world-model code."""

    num_frames: int
    frame_dt: float
    bodies_per_world: int
    q_per_world: int
    dof_per_world: int
    pelvis_body: int
    object_body: int
    wrist_bodies: np.ndarray  # (2,) [right, left] palm-frame parent bodies
    palm_local_pos: np.ndarray  # (2, 3)
    palm_local_quat: np.ndarray  # (2, 4) xyzw
    scalar_q_ids: np.ndarray  # (43,)
    scalar_dof_ids: np.ndarray  # (43,)
    default_q: np.ndarray  # (43,) scalar order (observation default)
    ref_root: np.ndarray  # (F, 7) pos + quat xyzw
    ref_q: np.ndarray  # (F, 43) scalar order
    ref_wrist_pos: np.ndarray  # (F, 2, 3) [right, left]
    ref_wrist_quat: np.ndarray  # (F, 2, 4)
    ref_obj_pos: np.ndarray  # (F, 3)
    ref_obj_quat: np.ndarray  # (F, 4)
    contact_labels: np.ndarray  # (F, 2) [left, right]
    sonic_slots: np.ndarray  # (29,) scalar slot of each SONIC joint
    finger_slots: np.ndarray  # (14,) scalar slot of each direct finger joint
    sonic_scale: np.ndarray  # (29,)
    sonic_default: np.ndarray  # (29,)
    ref_finger_target: np.ndarray  # (F, 14)
    soft_lower: np.ndarray  # (43,)
    soft_upper: np.ndarray  # (43,)
    obj_mass: float
    obj_inertia_eff: float
    obj_com: np.ndarray  # (3,) body-local COM
    obj_radius: float
    wrench_basis: np.ndarray  # (512, 6)
    cone_cos: np.ndarray
    cone_sin: np.ndarray
    contact_sides: tuple
    voc: dict
    objective: dict
    termination: dict
    latent_table: np.ndarray | None = None  # (F, 64) filled by sonic_jax
    finger_names: tuple = ()  # scalar joint name of each entry of finger_slots
    sonic_names: tuple = ()  # scalar joint name of each entry of sonic_slots (the first 29 action dimensions)


def extract_consts(handles: Handles) -> Consts:
    env = handles.env
    cfg = handles.cfg
    obs = env.observation_strategy
    action = env.action
    objective = env.objective
    model = env.model
    W = env.world_count
    F = handles.reference.num_frames
    ref_joint_q = _np(obs.reference.joint_q)  # (F, nq)
    scalar_q_ids = _np(obs.scalar_q_ids).astype(np.int64)
    sonic_slots = _np(action.sonic_scalar_slots).astype(np.int64)
    finger_slots = _np(action.finger_scalar_slots).astype(np.int64)
    finger_dof = _np(action.finger_dof_ids).astype(np.int64)
    obj_body = int(obs.object_body_id)
    bodies_per_world = model.body_count // W
    mass = float(_np(model.body_mass)[obj_body])
    inertia = _np(model.body_inertia).reshape(-1, 3, 3)[obj_body]
    com = _np(model.body_com).reshape(-1, 3)[obj_body]
    offsets = np.asarray(env.scene.object_root_position_offsets_w)
    if np.abs(offsets).max() > 1e-9:
        raise ValueError(f"object root offsets must be zero for this task, got max {np.abs(offsets).max()}")
    obj_cfg = cfg.objective
    basis = np.concatenate(
        [sample_wrench_basis(int(obj_cfg.num_wrench_basis), 1.0, int(obj_cfg.wrench_basis_seed))], axis=0
    )
    cone_cos, cone_sin = friction_cone_angles(int(obj_cfg.num_friction_cone_edges))
    wrist_pos = _np(obs.reference.wrist_pos_w).reshape(F, 2, 3)
    wrist_quat = _np(obs.reference.wrist_quat_w).reshape(F, 2, 4)
    # Objective uses (left, right); observation uses (right, left). Same frames, reversed order.
    obj_wrist = _np(objective.reference.wrist_pos_w).reshape(F, 2, 3)
    if not np.allclose(obj_wrist, wrist_pos[:, ::-1], atol=1e-6):
        raise ValueError("objective and observation wrist references are not mirror-ordered")
    return Consts(
        num_frames=F,
        frame_dt=float(env.frame_dt),
        bodies_per_world=bodies_per_world,
        q_per_world=model.joint_coord_count // W,
        dof_per_world=model.joint_dof_count // W,
        pelvis_body=int(obs.pelvis_body_id),
        object_body=obj_body,
        wrist_bodies=_np(obs.wrist_frames.body_ids).astype(np.int64),
        palm_local_pos=_np(obs.wrist_frames.body_to_frame_pos).reshape(2, 3),
        palm_local_quat=_np(obs.wrist_frames.body_to_frame_quat).reshape(2, 4),
        scalar_q_ids=scalar_q_ids,
        scalar_dof_ids=_np(obs.scalar_dof_ids).astype(np.int64),
        default_q=_np(obs.default_joint_position),
        ref_root=ref_joint_q[:, :7],
        ref_q=ref_joint_q[:, scalar_q_ids],
        ref_wrist_pos=wrist_pos,
        ref_wrist_quat=wrist_quat,
        ref_obj_pos=_np(obs.object_reference_pos_w).reshape(F, -1, 3)[:, 0],
        ref_obj_quat=_np(obs.object_reference_quat_w).reshape(F, -1, 4)[:, 0],
        contact_labels=_np(objective.contact_active).reshape(F, 2),
        sonic_slots=sonic_slots,
        finger_slots=finger_slots,
        finger_names=tuple(env.scene.layout.scalar_joints.names[i] for i in finger_slots),
        sonic_names=tuple(env.scene.layout.scalar_joints.names[i] for i in sonic_slots),
        sonic_scale=_np(action.sonic_scale),
        sonic_default=_np(action.default_sonic_position),
        ref_finger_target=_np(action.reference_joint_target)[:, finger_dof],
        soft_lower=_np(objective.soft_lower),
        soft_upper=_np(objective.soft_upper),
        obj_mass=mass,
        obj_inertia_eff=float(np.trace(inertia) / 3.0),
        obj_com=com,
        obj_radius=float(_np(env.command.object_radius)[0]),
        wrench_basis=basis,
        cone_cos=cone_cos,
        cone_sin=cone_sin,
        contact_sides=tuple(env.contact.layout.sides),
        voc=dict(  # the runtime VOC actually applied in this env (may differ from the official config)
            k_lin=float(env.config.voc.k_lin), d_lin=float(env.config.voc.d_lin), k_ang=float(env.config.voc.k_ang),
            d_ang=float(env.config.voc.d_ang), max_force=float(env.config.voc.max_force),
            max_torque=float(env.config.voc.max_torque), gravity=float(env.config.voc.gravity),
            mass_normalized=bool(env.config.voc.mass_normalized_gains),
        ),
        objective=dict(
            weights=np.asarray(objective.weights.numpy(), dtype=np.float32),
            pelvis_position_std=float(obj_cfg.pelvis_position_std),
            pelvis_orientation_std=float(obj_cfg.pelvis_orientation_std),
            body_joint_position_std=float(obj_cfg.body_joint_position_std),
            object_position_std=float(obj_cfg.object_position_std),
            palm_position_std=float(obj_cfg.palm_position_std),
            palm_orientation_std=float(obj_cfg.palm_orientation_std),
            force_closure_min_support=float(obj_cfg.force_closure_min_support),
            friction=float(obj_cfg.friction_coefficient),
        ),
        termination=dict(  # the runtime thresholds of this env (may be relaxed for training)
            pelvis_position=float(env.config.termination.pelvis_position_threshold),
            pelvis_orientation=float(env.config.termination.pelvis_orientation_threshold),
            palm_position=float(env.config.termination.palm_position_threshold),
            palm_orientation=float(env.config.termination.palm_orientation_threshold),
            object_position=float(env.config.termination.object_position_threshold),
            object_orientation=float(env.config.termination.object_orientation_threshold),
            freeze=int(env.config.termination.reset_freeze_steps),
        ),
    )


# ---------------------------------------------------------------------------------------------
# Zero-copy readout. All arrays are copied by the jitted packers below before the env mutates them.


class Readout:
    """Views of the live Warp buffers that the collector copies every step."""

    def __init__(self, env: RLEnv, consts: Consts) -> None:
        self.env = env
        self.c = consts
        W = env.world_count
        self.W = W
        a = env.action
        self.sonic_current = lambda: to_jax(a.current_observation, (W, 93))
        self.sonic_decoded = lambda: to_jax(a.decoder_action, (W, 29))
        self.processed_target = lambda: to_jax(a.processed_target, (W, 43))
        self.raw_action = lambda: to_jax(a.raw_action, (W, 43))
        self.target_history = lambda: to_jax(a.actor_action_history, (W, 3, 43))
        ct = env.contact
        S = ct.layout.slots_per_world
        self.contact_pos = lambda: wp.to_jax(ct.contact_pos_o).reshape(W, S, 3)
        self.contact_dir = lambda: wp.to_jax(ct.contact_force_direction_o).reshape(W, S, 3)
        self.contact_active = lambda: to_jax(ct.contact_active, (W, S))
        rp = env.reset_policy
        self.age = lambda: to_jax(rp.steps_since_reset, (W,))
        self.applied_voc = lambda: to_jax(rp.applied_voc_scale, (W,))
        self.timestep = lambda: to_jax(env.timestep, (W,))
        self.episode_step = lambda: to_jax(env.episode_step, (W,))
        self.start_frame = lambda: to_jax(env.episode_start_frame, (W,))
        self.terminated = lambda: to_jax(env.terminated, (W,))
        self.truncated = lambda: to_jax(env.truncated, (W,))
        self.objective_terms = lambda: to_jax(env.objective.terms, (W, 12))
        self.observation = lambda: to_jax(env.observation, (W, env.observation_dim))
        self.lags = lambda: jnp.stack([to_jax(d.delay_steps, (W,)) for d in a.delays], axis=-1)
        tm = env.termination
        self.term_causes = lambda: jnp.stack(
            [to_jax(tm.pelvis_done, (W,)), to_jax(tm.palm_done, (W,)), to_jax(tm.object_done, (W,))], axis=-1
        )

    def physics(self):
        s = self.env.state_0
        nb = self.c.bodies_per_world
        body_q = wp.to_jax(s.body_q).reshape(self.W, nb, 7)
        body_qd = wp.to_jax(s.body_qd).reshape(self.W, nb, 6)
        joint_q = to_jax(s.joint_q, (self.W, self.c.q_per_world))
        joint_qd = to_jax(s.joint_qd, (self.W, self.c.dof_per_world))
        return body_q, body_qd, joint_q, joint_qd


def sync() -> None:
    wp.synchronize_device()
