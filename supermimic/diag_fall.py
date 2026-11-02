"""Why do policies fall, and where does the object orientation error come from?

Evaluator-condition rollouts over many worlds. Falling worlds are aligned on their fall time and compared with the same
frames of worlds that stayed up: pelvis tilt, residual size per body part, joint tracking, support margin. The orientation
error of worlds that keep the object is split into tilt of the object's long axis and spin about it.
usage: python -m supermimic.diag_fall <policy.safetensors> [...]
"""

import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import numpy as np
import warp as wp

from flash_chord.training.flash_sac.evaluation import load_policy_for_inference

from supermimic.diag_eval_stats import WORLDS, build_eval_env

GROUPS = {"leg": ("hip", "knee", "ankle"), "waist": ("waist",), "arm": ("shoulder", "elbow", "wrist"), "finger": ("hand_",)}
OFFSETS = (-100, -80, -60, -40, -30, -20, -10, -5, 0, 10)


def quat_rotate(q, v):
    """xyzw quaternion(s) applied to vector(s)."""
    u, w = q[..., :3], q[..., 3:]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def rollout(env, jax_env, training, ck, F, nb, pelvis, feet, names):
    env.reset(frame_id=0)
    obs = jnp.asarray(env.observation.numpy().reshape(WORLDS, env.observation_dim))
    actor, act_fn, key, _ = load_policy_for_inference(ck, training, obs, env.action.action_dim, deterministic=True)
    log = {k: np.zeros((F, WORLDS)) for k in ("pz", "tilt", "pv", "pw", "margin", "track_leg", "rate")}
    log.update({f"a_{g}": np.zeros((F, WORLDS)) for g in GROUPS})
    action_abs = np.zeros((F, WORLDS, 43))
    bq = np.zeros((F, WORLDS, nb, 7), np.float32)
    idx = {g: [i for i, n in enumerate(names) if any(t in n for t in ts) and (g == "finger" or "hand_" not in n)] for g, ts in GROUPS.items()}
    prev = None
    for f in range(F):
        action, key = act_fn(actor, obs, key)
        a = np.asarray(action, np.float32).reshape(WORLDS, -1)
        obs = jax_env.step(action).observation
        b = env.state_0.body_q.numpy().reshape(WORLDS, nb, 7)
        v = env.state_0.body_qd.numpy().reshape(WORLDS, nb, 6)
        bq[f] = b
        pq = b[:, pelvis]
        log["pz"][f] = pq[:, 2]
        log["tilt"][f] = np.degrees(np.arccos(np.clip(quat_rotate(pq[:, 3:7], np.array([0.0, 0.0, 1.0]))[:, 2], -1, 1)))
        log["pv"][f] = np.linalg.norm(v[:, pelvis, :3], axis=-1)
        log["pw"][f] = np.linalg.norm(v[:, pelvis, 3:6], axis=-1)
        mid = 0.5 * (b[:, feet[0], :2] + b[:, feet[1], :2])
        log["margin"][f] = np.linalg.norm(pq[:, :2] - mid, axis=-1)
        for g, ids in idx.items():
            log[f"a_{g}"][f] = np.sqrt(np.mean(np.square(a[:, ids]), -1))
        action_abs[f] = np.abs(a)
        log["rate"][f] = 0.0 if prev is None else np.sqrt(np.mean(np.square(a - prev), -1))
        prev = a
    return log, action_abs, bq


def main(ckpts):
    with wp.ScopedDevice("cuda:0"):
        env, jax_env, setup, training = build_eval_env(ckpts[0])
        F = setup.reference.num_frames
        model = env.model
        nb = model.body_count // WORLDS
        labels = list(model.body_label) if hasattr(model, "body_label") else list(model.body_key)
        feet = [next(i for i, n in enumerate(labels[:nb]) if s in n) for s in ("left_ankle_roll", "right_ankle_roll")]
        pelvis = int(env.observation_strategy.pelvis_body_id)
        obj = int(env.command.layout.body_ids[0])
        scalar = list(env.scene.layout.scalar_joints.names)
        a_names = [scalar[i] for i in np.asarray(env.action.sonic_scalar_slots.numpy())] + [scalar[i] for i in np.asarray(env.action.finger_scalar_slots.numpy())]
        ref_p = np.asarray(setup.reference.object_body_pos_w(), np.float32)[:, 0]
        ref_q = np.asarray(setup.reference.object_body_quat_w(), np.float32)[:, 0][:, [1, 2, 3, 0]]
        # object's long axis from its collision/visual vertices (body frame)
        verts = []
        for i in range(model.shape_count):
            src = model.shape_source[i]
            if int(model.shape_body.numpy()[i]) == obj and src is not None and hasattr(src, "vertices"):
                verts.append(np.asarray(src.vertices) * model.shape_scale.numpy()[i])
        pts = np.concatenate(verts)
        evals, evecs = np.linalg.eigh(np.cov((pts - pts.mean(0)).T))
        axis = evecs[:, -1]
        step_deg = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(ref_q[1:] * ref_q[:-1], -1)), 0, 1)))
        print(f"object long axis (body frame) {np.round(axis, 2)}, principal std {np.round(np.sqrt(evals[::-1]), 3)} m; "
              f"reference orientation: median frame-to-frame {np.median(step_deg):.2f} deg, p99 {np.percentile(step_deg, 99):.2f}, total path {step_deg.sum():.0f} deg", flush=True)
        for ck in ckpts:
            print("==", "/".join(ck.split("/")[-3:]), flush=True)
            log, action_abs, bq = rollout(env, jax_env, training, ck, F, nb, pelvis, feet, a_names)
            pz = log["pz"]
            fall = np.array([next((i for i in range(F) if pz[i, w] < 0.5), F) for w in range(WORLDS)])
            fell = fall < F
            err = np.linalg.norm(bq[:, :, obj, :3] - ref_p[:, None], axis=-1)
            off = np.array([next((i for i in range(F - 9) if (err[i:i + 10, w] > 0.10).all()), F) for w in range(WORLDS)])
            print(f"fell {fell.sum()}/{WORLDS}; fall frame p10/50/90 {np.percentile(fall[fell], [10, 50, 90]).astype(int) if fell.any() else '-'}; "
                  f"fell while object still tracked {np.sum(fell & (off >= fall))}, after losing it {np.sum(fell & (off < fall))}", flush=True)
            foot_obj = np.minimum(*[np.linalg.norm(bq[:, :, feet[i], :2] - bq[:, :, obj, :2], axis=-1) for i in range(2)])  # (F, W)
            obj_on_floor = bq[:, :, obj, 2] < 0.2
            obj_move = np.zeros_like(foot_obj)
            obj_move[1:] = np.linalg.norm(np.diff(bq[:, :, obj, :2], axis=0), axis=-1) * 50.0
            lost = np.nonzero(off < F - 50)[0]
            up_lost = np.array([w for w in lost if not fell[w]], dtype=int)
            if len(up_lost):
                # lost-object worlds that stayed up: how close do the feet come, over the frames after the loss
                near = [foot_obj[off[w] + 20 :, w].min() if off[w] + 20 < F else np.nan for w in up_lost]
                print(f"lost the object and stayed up ({len(up_lost)} worlds): closest foot-object distance after the loss, median {np.nanmedian(near):.2f} m; "
                      f"share of those frames with a foot within 15 cm of a floor object: "
                      f"{np.mean([np.mean((foot_obj[off[w] + 20 :, w] < 0.15) & obj_on_floor[off[w] + 20 :, w]) if off[w] + 20 < F else 0 for w in up_lost]):.3f}", flush=True)
            if fell.any():
                w = np.nonzero(fell)[0]
                print("foot-object horizontal distance and object speed, aligned on the fall (falling worlds, median):", flush=True)
                for dt in (-100, -80, -60, -40, -30, -20, -10, -5, 0):
                    fr = np.clip(fall[w] + dt, 0, F - 1)
                    print(f"  t{dt:+4d}: foot-object {np.median(foot_obj[fr, w]):.2f} m, within 15 cm {np.mean(foot_obj[fr, w] < 0.15):.2f}, object on floor {np.mean(obj_on_floor[fr, w]):.2f}, object speed {np.median(obj_move[fr, w]):.2f} m/s (p90 {np.percentile(obj_move[fr, w], 90):.2f})", flush=True)
                touched = np.array([np.any((foot_obj[np.clip(fall[i] - 60, 0, F - 1) : fall[i], i] < 0.15) & obj_on_floor[np.clip(fall[i] - 60, 0, F - 1) : fall[i], i]) for i in w])
                print(f"  falls with a foot within 15 cm of the floor object in the 60 frames before: {touched.sum()}/{len(w)}", flush=True)
                print("aligned on the fall (median over falling worlds):", flush=True)
                print("  t-t0   frame  pelvis_z tilt_deg pel_v pel_w foot_margin | rms a: leg waist arm finger | a_rate", flush=True)
                for dt in OFFSETS:
                    fr = np.clip(fall[fell] + dt, 0, F - 1)
                    w = np.nonzero(fell)[0]
                    m = lambda k: np.median(log[k][fr, w])  # noqa: E731
                    print(f"  {dt:5d}  {np.median(fr):5.0f}  {m('pz'):6.2f}  {m('tilt'):6.1f}  {m('pv'):5.2f} {m('pw'):5.2f}  {m('margin'):6.2f}  | "
                          f"{m('a_leg'):5.2f} {m('a_waist'):5.2f} {m('a_arm'):5.2f} {m('a_finger'):5.2f} | {m('rate'):5.2f}", flush=True)
                if (~fell).any():
                    # same frames in the worlds that stayed up
                    w = np.nonzero(fell)[0]
                    fr0 = np.clip(fall[w] - 30, 0, F - 1)
                    up = np.nonzero(~fell)[0]
                    m = lambda k: np.median(log[k][fr0][:, up])  # noqa: E731
                    print(f"  stayed-up worlds at the same frames (t0-30): pelvis_z {m('pz'):.2f} tilt {m('tilt'):.1f} margin {m('margin'):.2f} | "
                          f"rms a: leg {m('a_leg'):.2f} waist {m('a_waist'):.2f} arm {m('a_arm'):.2f} finger {m('a_finger'):.2f} | rate {m('rate'):.2f}", flush=True)
                # which joints push hardest in the 40 frames before the fall
                w = np.nonzero(fell)[0]
                win = np.stack([action_abs[np.clip(fall[i] - 40, 0, F - 1) : max(fall[i] - 5, 1), i].mean(0) for i in w]).mean(0)
                order = np.argsort(win)[::-1][:8]
                print("  largest |a| in the 40 frames before falling:", ", ".join(f"{a_names[i].replace('_joint', '')} {win[i]:.1f}" for i in order), flush=True)
                print("  saturated (|a|>9) share before falling: %.3f" % float((np.stack([action_abs[np.clip(fall[i] - 40, 0, F - 1) : max(fall[i] - 5, 1), i].mean(0) > 9 for i in w])).mean()), flush=True)
            # orientation: tilt of the long axis vs spin about it, for worlds that keep the object
            good = np.nonzero(err.mean(0) < 0.10)[0]
            if len(good):
                sim_axis = quat_rotate(bq[:, good, obj, 3:7], axis.astype(np.float32))
                ref_axis = quat_rotate(ref_q[:, None], axis.astype(np.float32))
                tilt = np.degrees(np.arccos(np.clip(np.sum(sim_axis * ref_axis, -1), -1, 1)))
                d = np.abs(np.sum(bq[:, good, obj, 3:7] * ref_q[:, None], -1)) / np.linalg.norm(bq[:, good, obj, 3:7], axis=-1)
                total = np.degrees(2 * np.arccos(np.clip(d, 0, 1)))
                spin = np.sqrt(np.maximum(total**2 - tilt**2, 0.0))
                print(f"orientation in {len(good)} worlds with mean error < 10 cm (median deg): frame total tilt spin~", flush=True)
                for f in np.linspace(0, F - 1, 11).astype(int):
                    print(f"  {f:5d}  {np.median(total[f]):5.0f} {np.median(tilt[f]):5.0f} {np.median(spin[f]):5.0f}", flush=True)
                print(f"  mean over the episode: total {total.mean():.0f} tilt {tilt.mean():.0f} spin~ {spin.mean():.0f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
