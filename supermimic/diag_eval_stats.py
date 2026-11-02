"""Evaluator-condition statistics of checkpoints over many worlds (a single official evaluation is very noisy).

Environment built exactly as flash_chord.evaluation.harness does (checkpoint config, evaluation composition), one rollout per
world with the per-world delay draws. The object metrics follow the official ones: SP = mean error <= 10 cm and <= 0.5 rad,
MP = mean error < 3 cm and < 30 deg.
usage: python -m supermimic.diag_eval_stats <parquet (ignored, kept for the callers)> <policy.safetensors> [...]
"""

import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
from omegaconf import OmegaConf

from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.evaluation.harness import _evaluation_curriculum_stage, _resolve_parquet, apply_evaluation_composition
from flash_chord.evaluation.metrics import ADD_THRESHOLDS_M, add_auc, object_add, object_mppe_cm
from flash_chord.evaluation.rollout import sample_object_vertices
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import checkpoint_config
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.flash_sac.config import TrainingConfig
from flash_chord.training.flash_sac.evaluation import load_policy_for_inference
from flash_chord.training.flash_sac.network import Actor

WORLDS = 64


def q(x, ps=(10, 50, 90)):
    return "/".join(f"{np.percentile(x, p):.0f}" for p in ps)


def build_eval_env(ck, worlds=WORLDS):
    """Environment built exactly as the official harness does; call inside a warp device scope."""
    metadata = read_checkpoint_metadata(ck)
    config = checkpoint_config(metadata)
    apply_evaluation_composition(config, world_count=worlds, reset_mode="explicit")
    config["task"]["parquet"] = _resolve_parquet(config, None)
    cfg = OmegaConf.create(config)
    env_config = instantiate_typed(cfg.env, RLEnvConfig)
    training = instantiate_typed(cfg.training, TrainingConfig)
    setup = setup_scene(
        parquet=cfg.task.parquet, control_fps=env_config.sim.fps, motion_speed=cfg.task.motion_speed,
        embodiment=instantiate_typed(cfg.embodiment, Embodiment), collision=instantiate_typed(cfg.collision, CollisionPolicy),
        world_count=cfg.scene.world_count, include_support=cfg.scene.support, decompose_objects=cfg.scene.decompose_objects,
        object_scale_min=cfg.scene.get("object_scale_min", 1.0), object_scale_max=cfg.scene.get("object_scale_max", 1.0),
        object_scale_seed=cfg.scene.get("object_scale_seed", 0), contact_friction=cfg.scene.get("contact_friction", 1.0),
        object_free_joint_damping=cfg.scene.get("object_free_joint_damping", 0.0),
        source_frame_playback=bool(cfg.task.get("source_frame_playback", False)),
        motion_start_frame=int(cfg.task.get("motion_start_frame", 0)), motion_end_frame=int(cfg.task.get("motion_end_frame", -1)),
    )
    env = RLEnv(setup.scene, setup.reference, config=env_config)
    env.reset()
    env.capture_step()
    jax_env = WarpRLEnv(env, publish_diagnostics=True, publish_step_metrics=True)
    jax_env.set_curriculum(_evaluation_curriculum_stage(training.curriculum.stages[int(metadata["curriculum_stage"])], "explicit"))
    return env, jax_env, setup, training


def feed_forward(knots, knot_frames, frames):
    """Pre-tanh action offset per frame of a table with one knot every `knot_frames` frames."""
    x = np.arange(frames) / knot_frames
    i = np.floor(x).astype(int)
    w = (x - i)[:, None]
    return knots[i] * (1.0 - w) + knots[np.minimum(i + 1, len(knots) - 1)] * w


def main(ckpts):
    specs = [tuple((c.split("::") + [""])[:2]) for c in ckpts]  # <policy.safetensors>[::<feed-forward table.npz>]
    with wp.ScopedDevice("cuda:0"):
        env, jax_env, setup, training = build_eval_env(specs[0][0])
        F = setup.reference.num_frames
        nb = env.model.body_count // WORLDS
        obj, pelvis = int(env.command.layout.body_ids[0]), int(env.observation_strategy.pelvis_body_id)
        ref_p = np.asarray(setup.reference.object_body_pos_w(), np.float32)[:, 0]
        ref_q = np.asarray(setup.reference.object_body_quat_w(), np.float32)[:, 0][:, [1, 2, 3, 0]]
        ref_pose = np.concatenate([ref_p, ref_q], -1)[:, None].astype(np.float64)
        vertices = sample_object_vertices(env.scene)
        for spec in specs:
            ck, ff_path = spec
            env.reset(frame_id=0)
            obs = jnp.asarray(env.observation.numpy().reshape(WORLDS, env.observation_dim))
            actor, act_fn, key, _ = load_policy_for_inference(ck, training, obs, env.action.action_dim, deterministic=True)
            if ff_path:  # the policy plus a feed-forward table of pre-tanh action offsets (PLAN.md 13.22.6)
                table = np.load(ff_path)
                offsets = jnp.asarray(feed_forward(table[os.environ.get("FF_KEY", "ema_ff")], int(table["knot_frames"]), F), jnp.float32)
                module = Actor(action_dim=43, config=training.network, dtype=jnp.float32)
                variables = {"params": actor.params, "batch_stats": actor.batch_stats}
                plain = jax.jit(lambda o, d: jnp.tanh(module.apply(variables, o, training=False).mean + d) * 10.0)
            err, ang, pz, touch = (np.zeros((F, WORLDS)) for _ in range(4))
            pose = np.zeros((F, WORLDS, 1, 7))
            lags = np.stack([d.delay_steps.numpy() for d in env.action.delays], -1)
            for f in range(F):
                if ff_path:
                    action = plain(obs, offsets[f])
                else:
                    action, key = act_fn(actor, obs, key)
                obs = jax_env.step(action).observation
                b = env.state_0.body_q.numpy().reshape(WORLDS, nb, 7)
                pose[f, :, 0] = b[:, obj]
                err[f] = np.linalg.norm(b[:, obj, :3] - ref_p[f], axis=-1)
                d = np.abs(np.sum(b[:, obj, 3:7] * ref_q[f], -1)) / np.linalg.norm(b[:, obj, 3:7], axis=-1)
                ang[f] = 2 * np.arccos(np.clip(d, 0, 1))
                pz[f] = b[:, pelvis, 2]
                touch[f] = env.contact.contact_active.numpy().reshape(WORLDS, -1).sum(-1) > 0
            e, a = err.mean(0), ang.mean(0)
            off = np.array([next((i for i in range(F - 9) if (err[i:i + 10, w] > 0.10).all()), F) for w in range(WORLDS)])
            name = "/".join((ff_path or ck).split("/")[-3:]).replace("checkpoints/", "").replace(".safetensors", "").replace(".npz", "")
            print(f"{name[-40:]} err_cm {q(100 * e)} | deg {q(np.degrees(a))} | P(<3cm) {np.mean(e < 0.03):.2f} P(<5) {np.mean(e < 0.05):.2f} "
                  f"P(<10) {np.mean(e < 0.10):.2f} | SP {np.mean((e <= 0.10) & (a <= 0.5)):.2f} MP {np.mean((e < 0.03) & (a < np.radians(30))):.2f} | "
                  f"off@ {q(off)} | fell {np.mean(pz.min(0) < 0.4):.2f} touch {touch.mean():.2f}", flush=True)
            # official AUC and MPPE both use points on the object, so they also count its rotation; posAUC is the same curve on the centre alone
            add = object_add(pose, ref_pose, vertices)
            world_auc = [add_auc(add[:, [w]])[0] for w in range(WORLDS)]
            pos_auc = float(np.trapezoid([np.mean(err < t) for t in ADD_THRESHOLDS_M], x=np.linspace(0.0, 1.0, ADD_THRESHOLDS_M.size)))
            print(f"{name[-40:]} AUC {add_auc(add)[0]:.3f} (per world {'/'.join(f'{np.percentile(world_auc, p):.2f}' for p in (10, 50, 90))}) "
                  f"posAUC {pos_auc:.3f} MPPE {object_mppe_cm(pose, ref_pose):.2f}", flush=True)
            if os.environ.get("FALL_DETAIL"):
                fall = np.array([next((i for i in range(F) if pz[i, w] < 0.5), F) for w in range(WORLDS)])
                fell, good = fall < F, e < 0.10
                print(f"  fell {fell.sum()}/{WORLDS}: fall frame {q(fall[fell]) if fell.any() else '-'}; object already off first in {np.sum(fell & (off < fall))}, "
                      f"fell while still tracking {np.sum(fell & (off >= fall))}; median gap fall-off {np.median((fall - off)[fell]) if fell.any() else 0:.0f}", flush=True)
                print(f"  object lost without falling {np.sum(~fell & ~good)}, tracked <10 cm and upright {np.sum(~fell & good)}, tracked <10 cm but fell {np.sum(fell & good)}", flush=True)
                print(f"  mean lag per delay group: good {np.round(lags[good].mean(0), 2) if good.any() else '-'} bad {np.round(lags[~good].mean(0), 2)}", flush=True)
                print(f"  off frame histogram (bins of 50): {np.histogram(off, bins=np.arange(0, F + 51, 50))[0].tolist()}", flush=True)
                print(f"  fall frame histogram (bins of 50): {np.histogram(fall[fell], bins=np.arange(0, F + 51, 50))[0].tolist()}", flush=True)


if __name__ == "__main__":
    main(sys.argv[2:])
