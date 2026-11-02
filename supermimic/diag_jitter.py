"""How much does each policy shake? Evaluation-condition rollouts of several checkpoints of one episode.

For every policy: action rate and saturation, the share of action energy above 5 Hz, and the wrist and pelvis motion above 5 Hz that the
reference does not have. Run on the frames before the object is released, on worlds that did not fall.
usage: python -m supermimic.diag_jitter <label=policy.safetensors> [...]
"""

import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import numpy as np
import warp as wp

from flash_chord.training.flash_sac.evaluation import load_policy_for_inference

from supermimic.diag_eval_stats import build_eval_env

WORLDS = 16
FPS = 50.0


def high_share(x, cut=5.0):
    """Share of the signal's energy above `cut` Hz, per time axis 0 (mean removed)."""
    spec = np.abs(np.fft.rfft(x - x.mean(0, keepdims=True), axis=0)) ** 2
    freq = np.fft.rfftfreq(x.shape[0], 1.0 / FPS)
    return float(spec[freq >= cut].sum() / max(spec[freq > 0].sum(), 1e-12))


def high_rms_mm(x, cut=5.0):
    """RMS amplitude in mm of the part of a position signal above `cut` Hz, averaged over worlds and bodies."""
    spec = np.fft.rfft(x - x.mean(0, keepdims=True), axis=0)
    freq = np.fft.rfftfreq(x.shape[0], 1.0 / FPS)
    spec[freq < cut] = 0
    hp = np.fft.irfft(spec, n=x.shape[0], axis=0)
    return float(1000.0 * np.sqrt(np.mean(np.sum(hp**2, axis=-1))))


def main(items):
    print("policy | fell | |action| mean, sat>9 | action rate body, fingers (raw /step) | action energy >5 Hz body, fingers | >5 Hz RMS mm: wrist, pelvis, object | peak action Hz")
    for label, ck in items:
        with wp.ScopedDevice("cuda:0"):
            env, jax_env, setup, training = build_eval_env(ck, WORLDS)
            F = setup.reference.num_frames
            nb = env.model.body_count // WORLDS
            obj, pelvis = int(env.command.layout.body_ids[0]), int(env.observation_strategy.pelvis_body_id)
            wrists = np.asarray(env.observation_strategy.wrist_frames.body_ids.numpy())[:2]
            env.reset(frame_id=0)
            obs = jnp.asarray(env.observation.numpy().reshape(WORLDS, env.observation_dim))
            actor, act_fn, key, _ = load_policy_for_inference(ck, training, obs, env.action.action_dim, deterministic=True)
            act = np.zeros((F, WORLDS, 43), np.float32)
            bq = np.zeros((F, WORLDS, nb, 3), np.float32)
            for f in range(F):
                action, key = act_fn(actor, obs, key)
                act[f] = np.asarray(action).reshape(WORLDS, -1)
                obs = jax_env.step(action).observation
                bq[f] = env.state_0.body_q.numpy().reshape(WORLDS, nb, 7)[..., :3]
        fell = bq[:, :, pelvis, 2].min(0) < 0.4
        keep = ~fell if (~fell).any() else np.ones(WORLDS, bool)
        a, p = act[:, keep], bq[:, keep]
        rate = np.abs(np.diff(a, axis=0))
        ww = p[:, :, wrists]
        peak = np.fft.rfftfreq(a.shape[0], 1 / FPS)[1:][np.argmax((np.abs(np.fft.rfft(a[:, :, :29] - a[:, :, :29].mean(0), axis=0)) ** 2).mean((1, 2))[1:])]
        print(f"{label:22s} | {fell.mean():.2f} | {np.abs(a).mean():5.2f} {np.mean(np.abs(a) > 9):.2f} | {rate[..., :29].mean():.3f} {rate[..., 29:].mean():.3f} | "
              f"{high_share(a[..., :29]):.2f} {high_share(a[..., 29:]):.2f} | {high_rms_mm(ww):5.1f} {high_rms_mm(p[:, :, [pelvis]]):5.1f} {high_rms_mm(p[:, :, [obj]]):5.1f} | {peak:.1f}", flush=True)


if __name__ == "__main__":
    main([tuple(x.split("=", 1)) for x in sys.argv[1:]])
