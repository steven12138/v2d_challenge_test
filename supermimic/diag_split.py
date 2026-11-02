"""What separates worlds that keep the object from worlds that lose it?

Evaluator-condition rollouts of one checkpoint over many worlds with identical policy and start; only the per-world delay draws
(and numerics) differ. Prints, per frame around the interaction, the median of each signal in successful and failed worlds and
how well it predicts the outcome (rank AUC, 0.5 = no information), plus the same for the delay draws.
usage: python -m supermimic.diag_split <policy.safetensors> [first_frame last_frame step]
"""

import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import numpy as np
import warp as wp

from flash_chord.training.flash_sac.evaluation import load_policy_for_inference

from supermimic.diag_eval_stats import build_eval_env

WORLDS = 256


def rank_auc(x, good):
    """P(signal in a good world > signal in a bad world)."""
    a, b = x[good], x[~good]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float((np.sum(a[:, None] > b[None]) + 0.5 * np.sum(a[:, None] == b[None])) / (len(a) * len(b)))


def main(ck, first=40, last=160, step=10):
    with wp.ScopedDevice("cuda:0"):
        env, jax_env, setup, training = build_eval_env(ck, WORLDS)
        F = setup.reference.num_frames
        nb = env.model.body_count // WORLDS
        obj, pelvis = int(env.command.layout.body_ids[0]), int(env.observation_strategy.pelvis_body_id)
        wrists = np.asarray(env.observation_strategy.wrist_frames.body_ids.numpy())[:2]  # [right, left]
        ref_p = np.asarray(setup.reference.object_body_pos_w(), np.float32)[:, 0]
        ref_joint = np.asarray(env.observation_strategy.reference.joint_q.numpy())
        ref_wrist = np.asarray(env.observation_strategy.reference.wrist_pos_w.numpy()).reshape(F, 2, 3)
        env.reset(frame_id=0)
        obs = jnp.asarray(env.observation.numpy().reshape(WORLDS, env.observation_dim))
        lags = np.stack([d.delay_steps.numpy() for d in env.action.delays], -1).astype(float)
        actor, act_fn, key, _ = load_policy_for_inference(ck, training, obs, env.action.action_dim, deterministic=True)
        err = np.zeros((F, WORLDS))
        sig = {k: np.zeros((F, WORLDS)) for k in ("wrist_R", "wrist_L", "wrist_R_obj", "pelvis_xy", "pelvis_z", "pelvis_yaw_rate", "obj_disp", "contact_R", "contact_L", "act_rms")}
        p0 = None
        for f in range(F):
            action, key = act_fn(actor, obs, key)
            a = np.asarray(action, np.float32).reshape(WORLDS, -1)
            obs = jax_env.step(action).observation
            b = env.state_0.body_q.numpy().reshape(WORLDS, nb, 7)
            v = env.state_0.body_qd.numpy().reshape(WORLDS, nb, 6)
            if p0 is None:
                p0 = b[:, obj, :3].copy()
            err[f] = np.linalg.norm(b[:, obj, :3] - ref_p[f], axis=-1)
            sig["wrist_R"][f] = np.linalg.norm(b[:, wrists[0], :3] - ref_wrist[f, 0], axis=-1)
            sig["wrist_L"][f] = np.linalg.norm(b[:, wrists[1], :3] - ref_wrist[f, 1], axis=-1)
            sig["wrist_R_obj"][f] = np.linalg.norm(b[:, wrists[0], :3] - b[:, obj, :3], axis=-1)
            sig["pelvis_xy"][f] = np.linalg.norm(b[:, pelvis, :2] - ref_joint[f, :2], axis=-1)
            sig["pelvis_z"][f] = b[:, pelvis, 2]
            sig["pelvis_yaw_rate"][f] = np.abs(v[:, pelvis, 5])
            sig["obj_disp"][f] = np.linalg.norm(b[:, obj, :3] - p0, axis=-1)
            c = env.contact.contact_active.numpy().reshape(WORLDS, 2, 8).sum(-1)
            sig["contact_L"][f], sig["contact_R"][f] = c[:, 0], c[:, 1]
            sig["act_rms"][f] = np.sqrt(np.mean(np.square(a), -1))
        mean_err = err.mean(0)
        good = mean_err < 0.10
        print(f"{ck.split('/')[-1]}: {good.sum()}/{WORLDS} worlds with mean object error < 10 cm; median error good {100 * np.median(mean_err[good]):.1f} bad {100 * np.median(mean_err[~good]):.1f} cm", flush=True)
        off = np.array([next((i for i in range(F - 9) if (err[i:i + 10, w] > 0.10).all()), F) for w in range(WORLDS)])
        print("object first stays >10 cm off, histogram per 25 frames from 0:", np.histogram(off[~good], bins=np.arange(0, 301, 25))[0].tolist(), "later", int(np.sum(off[~good] >= 300)), flush=True)
        print("delay draws: mean lag per group good", np.round(lags[good].mean(0), 2), "bad", np.round(lags[~good].mean(0), 2),
              "| rank AUC per group", [round(rank_auc(lags[:, i], good), 2) for i in range(lags.shape[1])], flush=True)
        print("frame | " + " | ".join(f"{k}: good/bad/AUC" for k in sig), flush=True)
        for f in range(first, min(last, F - 1) + 1, step):
            cells = []
            for k, x in sig.items():
                s = 100.0 if k in ("wrist_R", "wrist_L", "wrist_R_obj", "pelvis_xy", "obj_disp") else 1.0
                cells.append(f"{s * np.median(x[f, good]):.1f}/{s * np.median(x[f, ~good]):.1f}/{rank_auc(x[f], good):.2f}")
            print(f"{f:4d} | " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], *(int(x) for x in sys.argv[2:5]))
