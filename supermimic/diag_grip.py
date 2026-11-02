"""Diagnostic: can a scripted finger closure hold the object without VOC?

Official settle reset at the given frames with the hand(s) the reference grasps with commanded to a closed pose; body from
the policy (or pure SONIC). Prints object tracking and contact after the release, per closure fraction.
Also prints the raw human finger curl around the grasp.
usage: python -m supermimic.diag_grip <parquet> <raw_episode.parquet> <policy.safetensors> <frame> [<frame> ...]
"""

import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq
import warp as wp

from flash_chord.training.flash_sac.evaluation import load_policy_for_inference

from supermimic import core, env_io

WORLDS = 32
FREE = {k: 99.0 for k in ("pelvis_position_threshold", "palm_position_threshold", "object_position_threshold",
                          "object_orientation_threshold", "palm_orientation_threshold", "pelvis_orientation_threshold")}
REPORT = (0, 10, 20, 40, 60, 100)


def grip_pose(names, lo, hi, frac):
    """Closed pose per finger joint: `frac` of the way to the farther limit; the thumb base rotation stays centred."""
    q = np.zeros(len(names), np.float32)
    for i, n in enumerate(names):
        if "thumb_0" not in n:
            q[i] = frac * (lo[i] if abs(lo[i]) > abs(hi[i]) else hi[i])
    return q


def human_curl(raw, names):
    t = pq.read_table(raw)
    pose = np.asarray(t["observation.human.pose"].to_pylist(), np.float32)
    meta = pq.read_schema(raw)
    return pose


def main(parquet, raw, ckpt, frames):
    with wp.ScopedDevice("cuda:0"):
        h = env_io.build(parquet, WORLDS, termination_overrides=FREE)
        env, tm = h.env, h.env.termination
        c = env_io.extract_consts(h)
        env.set_voc_scale(0.0)
        scalar = list(env.scene.layout.scalar_joints.names)
        names = [scalar[i] for i in c.finger_slots]
        lo, hi = c.soft_lower[c.finger_slots], c.soft_upper[c.finger_slots]
        lab = core.grasp_labels(c)
        obs = jnp.asarray(env.observation.numpy().reshape(WORLDS, env.observation_dim))
        actor, act_fn, key, _ = load_policy_for_inference(ckpt, h.training, obs, 43, deterministic=True)
        ref = np.asarray(h.reference.object_body_pos_w(), np.float32)[:, 0]
        for frame in frames:
            hands = [s for s, on in zip(("left", "right"), lab[frame]) if on > 0]
            mask = np.array([any(n.startswith(s) for s in hands) for n in names])
            print(f"== frame {frame} object z {ref[frame, 2]:.2f} grasping {hands}", flush=True)
            for body in ("sonic", "policy"):
                for frac in (0.0, 0.4, 0.7, 1.0):
                    env.reset_policy.start_frame = frame
                    env.reset_policy.end_frame = frame
                    env.reset(reset_mask=wp.ones(WORLDS, dtype=wp.int32))
                    target = grip_pose(names, lo, hi, frac) * mask
                    out = []
                    for step in range(51 + max(REPORT)):
                        o = jnp.asarray(env.observation.numpy().reshape(WORLDS, env.observation_dim))
                        a = np.array(act_fn(actor, o, key)[0], np.float32)
                        if body == "sonic":
                            a[:, :core.SONIC_DIM] = 0.0
                        ramp = min(step / 30.0, 1.0)
                        a[:, core.SONIC_DIM:] = np.where(mask, ramp * target / 0.15, a[:, core.SONIC_DIM:] if frac == 0.0 else 0.0)
                        env.step(wp.array(a.reshape(-1)))
                        wp.synchronize_device()
                        if step - 50 in REPORT:
                            act = env.contact.contact_active.numpy().reshape(WORLDS, 2, 8).sum(-1)
                            out.append(f"+{step - 50:<3d} {100 * np.median(tm.object_position_error.numpy()):5.1f}cm "
                                       f"L{act[:, 0].mean():.1f} R{act[:, 1].mean():.1f}")
                    print(f"  {body:6s} close {frac:.1f}: " + " | ".join(out), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4:]])
