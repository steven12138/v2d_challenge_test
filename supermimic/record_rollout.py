"""Record one evaluator-identical rollout of a checkpoint for rendering.

Builds the environment exactly as `flash_chord.evaluation.harness.evaluate_checkpoint` does (checkpoint config,
evaluation composition: 1 world, explicit frame-0 reset, VOC off, termination off, deterministic policy), then
records every body pose per control step plus the reference object pose, and exports the visual meshes.

usage: python -m supermimic.record_rollout <policy.safetensors> <out.npz> [worlds]
With several worlds (per-world delay draws, single evaluations vary a lot) the world with the smallest object error is kept.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import numpy as np
import warp as wp
from omegaconf import OmegaConf

from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.evaluation.harness import _evaluation_curriculum_stage, _resolve_parquet, apply_evaluation_composition
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import checkpoint_config
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.flash_sac.config import TrainingConfig
from flash_chord.training.flash_sac.evaluation import load_policy_for_inference

VISIBLE_ONLY = 1  # newton.ShapeFlags.VISIBLE without collision bits: render meshes


def main(checkpoint: str, out: str, worlds: int = 1) -> None:
    metadata = read_checkpoint_metadata(checkpoint)
    config = checkpoint_config(metadata)
    apply_evaluation_composition(config, world_count=worlds, reset_mode="explicit")
    config["task"]["parquet"] = _resolve_parquet(config, None)
    cfg = OmegaConf.create(config)
    env_config = instantiate_typed(cfg.env, RLEnvConfig)
    training = instantiate_typed(cfg.training, TrainingConfig)
    with wp.ScopedDevice("cuda:0"):
        setup = setup_scene(
            parquet=cfg.task.parquet,
            control_fps=env_config.sim.fps,
            motion_speed=cfg.task.motion_speed,
            embodiment=instantiate_typed(cfg.embodiment, Embodiment),
            collision=instantiate_typed(cfg.collision, CollisionPolicy),
            world_count=cfg.scene.world_count,
            include_support=cfg.scene.support,
            decompose_objects=cfg.scene.decompose_objects,
            object_scale_min=cfg.scene.get("object_scale_min", 1.0),
            object_scale_max=cfg.scene.get("object_scale_max", 1.0),
            object_scale_seed=cfg.scene.get("object_scale_seed", 0),
            contact_friction=cfg.scene.get("contact_friction", 1.0),
            object_free_joint_damping=cfg.scene.get("object_free_joint_damping", 0.0),
            source_frame_playback=bool(cfg.task.get("source_frame_playback", False)),
            motion_start_frame=int(cfg.task.get("motion_start_frame", 0)),
            motion_end_frame=int(cfg.task.get("motion_end_frame", -1)),
        )
        env = RLEnv(setup.scene, setup.reference, config=env_config)
        env.reset()
        env.capture_step()
        jax_env = WarpRLEnv(env, publish_diagnostics=True, publish_step_metrics=True)
        stage = int(metadata["curriculum_stage"])
        jax_env.set_curriculum(_evaluation_curriculum_stage(training.curriculum.stages[stage], "explicit"))
        env.reset(frame_id=0)
        obs = jnp.asarray(env.observation.numpy().reshape(worlds, env.observation_dim))
        actor, policy_action, key, _ = load_policy_for_inference(
            checkpoint, training, obs, env.action.action_dim, deterministic=True
        )
        steps = setup.reference.num_frames
        nb = env.model.body_count // worlds
        body_q = np.zeros((steps + 1, worlds, nb, 7), np.float32)
        body_q[0] = env.state_0.body_q.numpy().reshape(worlds, nb, 7)
        contact = np.zeros((steps, worlds, env.contact.layout.slots_per_world), np.float32)  # [left x8, right x8]
        for step in range(steps):
            action, key = policy_action(actor, obs, key)
            obs = jax_env.step(action).observation
            body_q[step + 1] = env.state_0.body_q.numpy().reshape(worlds, nb, 7)
            contact[step] = env.contact.contact_active.numpy().reshape(worlds, -1)
            if step % 100 == 0:
                print(f"step {step}/{steps}", flush=True)

        # Reset to each retargeted reference frame to expose the G1 motion used by the policy.
        # This runs only after the evaluated rollout has been recorded.
        ref_body_q = np.zeros((steps, nb, 7), np.float32)
        for frame in range(steps):
            env.reset(frame_id=frame)
            ref_body_q[frame] = env.state_0.body_q.numpy().reshape(worlds, nb, 7)[0]

        model = env.model
        flags = model.shape_flags.numpy()
        meshes = {}
        for i in range(model.shape_count):
            src = model.shape_source[i]
            if flags[i] != VISIBLE_ONLY or src is None or not hasattr(src, "vertices") or model.shape_body.numpy()[i] >= nb:
                continue
            meshes[f"mesh{i}_v"] = np.asarray(src.vertices, np.float32) * model.shape_scale.numpy()[i]
            meshes[f"mesh{i}_f"] = np.asarray(src.indices, np.int32).reshape(-1, 3)
            meshes[f"mesh{i}_body"] = np.int32(model.shape_body.numpy()[i])
            meshes[f"mesh{i}_tf"] = model.shape_transform.numpy()[i].astype(np.float32)
        ref_pos = np.asarray(setup.reference.object_body_pos_w(), np.float32)[:, 0]
        ref_quat_wxyz = np.asarray(setup.reference.object_body_quat_w(), np.float32)[:, 0]
        obj = int(env.command.layout.body_ids[0])
        err = np.linalg.norm(body_q[1:, :, obj, :3] - ref_pos[:, None], axis=-1).mean(0)
        best = int(np.argmin(err))
        pelvis_id = int(env.observation_strategy.pelvis_body_id)
        if os.environ.get("PICK") == "fall":  # a representative falling world: median fall frame
            pz = body_q[1:, :, pelvis_id, 2]
            fall = np.array([next((i for i in range(steps) if pz[i, w] < 0.5), steps) for w in range(worlds)])
            fallen = np.nonzero(fall < steps)[0]
            best = int(fallen[np.argsort(fall[fallen])[len(fallen) // 2]]) if len(fallen) else best
            print(f"picked falling world {best}: fall frame {fall[best]}, {len(fallen)}/{worlds} fell", flush=True)
        print(f"object error per world (cm): best {100 * err[best]:.1f} median {100 * np.median(err):.1f} worst {100 * err.max():.1f}", flush=True)
        np.savez_compressed(
            out,
            body_q=body_q[:, best],
            ref_body_q=ref_body_q,
            contact=contact[:, best],
            wrist_bodies=np.asarray(env.observation_strategy.wrist_frames.body_ids.numpy(), np.int32)[:2],
            object_body=np.int32(env.command.layout.body_ids[0]),
            pelvis_body=np.int32(env.observation_strategy.pelvis_body_id),
            ref_obj_pos=ref_pos,
            ref_obj_quat_xyzw=ref_quat_wxyz[:, [1, 2, 3, 0]],
            fps=np.float32(30.0),
            **meshes,
        )
        print(f"wrote {out}: {steps} steps, {len([k for k in meshes if k.endswith('_v')])} meshes", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
