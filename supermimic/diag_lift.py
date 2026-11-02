"""Diagnostic: frame-0 rollouts through the lift phase, with and without full VOC (gravity compensation on).

Prints palm / object tracking and hand-object contact counts every few frames, for pure SONIC and for a policy.
usage: python -m supermimic.diag_lift <parquet> <policy.safetensors> <first_frame> <last_frame>
"""

import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import numpy as np
import warp as wp

from flash_chord.training.flash_sac.evaluation import load_policy_for_inference

from supermimic import env_io

NO_TERMINATION = {"pelvis_position_threshold": 99.0, "palm_position_threshold": 99.0,
                  "object_position_threshold": 99.0, "object_orientation_threshold": 99.0,
                  "palm_orientation_threshold": 99.0, "pelvis_orientation_threshold": 99.0}


def rollout(handles, policy, voc, first, last, every=5):
    env = handles.env
    tm = env.termination
    env.set_voc_scale(voc)
    env.reset(frame_id=0)
    zero = wp.zeros(env.action_input.shape, dtype=wp.float32)
    ref_z = np.asarray(handles.reference.object_body_pos_w(), np.float32)[:, 0, 2]
    obj_body = int(env.observation_strategy.object_body_id)
    for step in range(min(last, handles.reference.num_frames - 1)):
        if policy is None:
            env.step(zero)
        else:
            obs = jnp.asarray(env.observation.numpy().reshape(1, env.observation_dim))
            action, _ = policy(obs)
            env.step(wp.array(np.asarray(action, np.float32).reshape(-1)))
        wp.synchronize_device()
        if step >= first and step % every == 0:
            act = env.contact.contact_active.numpy().reshape(-1)
            z = float(env.state_0.body_q.numpy()[obj_body, 2])
            print(f"  f{step:4d} palm {100 * tm.palm_position_error.numpy()[0]:5.1f}cm  "
                  f"obj {100 * tm.object_position_error.numpy()[0]:5.1f}cm/{np.degrees(tm.object_orientation_error.numpy()[0]):5.1f}deg  "
                  f"z {z:.2f} (ref {ref_z[step]:.2f})  contacts L{int(act[:8].sum())} R{int(act[8:].sum())}", flush=True)


def main(parquet, ckpt, first, last):
    for gravity in (False, True):
        with wp.ScopedDevice("cuda:0"):
            handles = env_io.build(parquet, 1, voc_gravity_comp=gravity, termination_overrides=NO_TERMINATION)
            env = handles.env
            obs = jnp.asarray(env.observation.numpy().reshape(1, env.observation_dim))
            actor, act_fn, key, _ = load_policy_for_inference(ckpt, handles.training, obs, 43, deterministic=True)

            def policy(o, _state={"key": key}):
                a, _state["key"] = act_fn(actor, o, _state["key"])
                return a, None

            cases = [("VOC=0", 0.0)] if not gravity else [("VOC=1 + gravity comp", 1.0)]
            for name, voc in cases:
                for pname, pol in (("SONIC", None), ("policy", policy)):
                    print(f"== {pname}, {name}", flush=True)
                    rollout(handles, pol, voc, first, last)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
