"""Diagnostic: does a settled grasp hold without VOC?

Official sampled reset at chosen frames (50-step settle: fingers close under full VOC while the frame is frozen), then
VOC fades to 0 and the frame advances. Prints object tracking and hand contact after the release, for pure SONIC and a policy.
usage: python -m supermimic.diag_hold <parquet> <policy.safetensors> <frame> [<frame> ...]
"""

import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import numpy as np
import warp as wp

from flash_chord.training.flash_sac.evaluation import load_policy_for_inference

from supermimic import env_io

WORLDS = 32
NO_TERMINATION = {"pelvis_position_threshold": 99.0, "palm_position_threshold": 99.0,
                  "object_position_threshold": 99.0, "object_orientation_threshold": 99.0,
                  "palm_orientation_threshold": 99.0, "pelvis_orientation_threshold": 99.0}
REPORT = (0, 5, 10, 20, 40, 60)  # steps after the 50-step settle


def rollout(handles, policy, frame):
    env = handles.env
    tm = env.termination
    env.reset_policy.start_frame = frame
    env.reset_policy.end_frame = frame
    env.reset(reset_mask=wp.ones(WORLDS, dtype=wp.int32))
    zero = wp.zeros(env.action_input.shape, dtype=wp.float32)
    out = []
    for step in range(51 + max(REPORT)):
        if policy is None:
            env.step(zero)
        else:
            obs = jnp.asarray(env.observation.numpy().reshape(WORLDS, env.observation_dim))
            env.step(wp.array(np.asarray(policy(obs), np.float32).reshape(-1)))
        wp.synchronize_device()
        if step - 50 in REPORT:
            act = env.contact.contact_active.numpy().reshape(WORLDS, 2, 8).sum(-1)
            out.append(f"+{step - 50:<2d} obj {100 * np.median(tm.object_position_error.numpy()):5.1f}cm "
                       f"palm {100 * np.median(tm.palm_position_error.numpy()):4.0f}cm "
                       f"L{act[:, 0].mean():.1f} R{act[:, 1].mean():.1f}")
    return " | ".join(out)


def main(parquet, ckpt, frames):
    with wp.ScopedDevice("cuda:0"):
        handles = env_io.build(parquet, WORLDS, termination_overrides=NO_TERMINATION)
        env = handles.env
        env.set_voc_scale(0.0)
        obs = jnp.asarray(env.observation.numpy().reshape(WORLDS, env.observation_dim))
        actor, act_fn, key, _ = load_policy_for_inference(ckpt, handles.training, obs, 43, deterministic=True)
        ref = np.asarray(handles.reference.object_body_pos_w(), np.float32)[:, 0]
        for frame in frames:
            print(f"== frame {frame} (reference object z {ref[frame, 2]:.2f})", flush=True)
            print("  SONIC :", rollout(handles, None, frame), flush=True)
            print("  policy:", rollout(handles, lambda o: act_fn(actor, o, key)[0], frame), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]])
