"""JAX re-implementation of the ReconBody observation, SONIC prior, objective and termination.

Every function mirrors a specific FlashCHORD kernel (file references in comments) and is checked
against the live environment at startup (`main.validate_against_env`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from supermimic.geometry import (
    geodesic_angle,
    log_map,
    mat_to_6d,
    mm,
    mT,
    mv,
    quat_to_mat,
    rot_angle_sq,
    rot_z,
    six_d_to_mat,
    yaw_of,
)

FUTURE_OFFSETS = (0, 5, 10)
N_JOINTS = 43
SONIC_DIM = 29
N_FINGERS = 14
HIST = 10
OBS_DIM = 486
TERM_NAMES = (
    "termination", "action_rate_l2", "action_l2", "soft_joint_limit", "pelvis_position", "pelvis_orientation",
    "body_joint_position", "object_position", "progress", "palm_position", "palm_orientation", "force_closure",
)


def reference_velocity(pos, dt, half=2):
    """Central difference over +-half frames (the mocap object track is noisy frame to frame)."""
    F = pos.shape[0]
    i = np.arange(F)
    lo, hi = np.maximum(i - half, 0), np.minimum(i + half, F - 1)
    return (pos[hi] - pos[lo]) / (np.maximum(hi - lo, 1)[:, None] * dt)


def grasp_labels(c, reach=0.35, lift=0.15):
    """Hands [left, right] that hold the object in the reference: labelled (or the object is off the ground,
    so something must hold it) and the reference palm is near the object. Some sequences label contact a
    second before the hands get there, others drop the labels while carrying (ep 17, frames 167-278)."""
    obj = np.asarray(c.ref_obj_pos)
    near = np.linalg.norm(np.asarray(c.ref_wrist_pos) - obj[:, None], axis=-1) < reach
    lifted = (obj[:, 2] > obj[0, 2] + lift)[:, None]
    return np.maximum(np.asarray(c.contact_labels), lifted) * near[:, ::-1]  # wrist order is [right, left]


def device_consts(c) -> dict:
    """Numpy task constants -> jnp pytree (static ints kept as python ints)."""
    f = lambda x: jnp.asarray(np.asarray(x), dtype=jnp.float32)  # noqa: E731
    obj_z = np.asarray(c.ref_obj_pos)[:, 2]
    lifted = np.flatnonzero(obj_z > obj_z[0] + 0.15)
    root_R = quat_to_mat(f(c.ref_root[:, 3:7]))
    idx = np.arange(c.num_frames)
    lo, hi = np.maximum(idx - 2, 0), np.minimum(idx + 2, c.num_frames - 1)
    root_w = log_map(mm(root_R[hi], mT(root_R[lo]))) / f(np.maximum(hi - lo, 1)[:, None] * c.frame_dt)
    return dict(
        ref_ov=f(reference_velocity(np.asarray(c.ref_obj_pos), c.frame_dt)),
        grasp_lab=f(grasp_labels(c)),
        F=int(c.num_frames),
        first_lift_frame=int(lifted[0]) if len(lifted) else int(c.num_frames),
        pelvis=int(c.pelvis_body),
        obj=int(c.object_body),
        wrist=np.asarray(c.wrist_bodies, dtype=np.int64),
        q_ids=np.asarray(c.scalar_q_ids, dtype=np.int64),
        dof_ids=np.asarray(c.scalar_dof_ids, dtype=np.int64),
        palm_lp=f(c.palm_local_pos),
        palm_lR=quat_to_mat(f(c.palm_local_quat)),
        default_q=f(c.default_q),
        ref_root_p=f(c.ref_root[:, :3]),
        ref_root_v=f(reference_velocity(np.asarray(c.ref_root)[:, :3], c.frame_dt)),
        ref_root_R=root_R,
        ref_root_w=root_w,
        ref_q=f(c.ref_q),
        ref_qd=f(reference_velocity(np.asarray(c.ref_q), c.frame_dt)),
        ref_wp=f(c.ref_wrist_pos),
        ref_wR=quat_to_mat(f(c.ref_wrist_quat)),
        ref_op=f(c.ref_obj_pos),
        ref_oR=quat_to_mat(f(c.ref_obj_quat)),
        labels=f(c.contact_labels),
        sonic_slots=np.asarray(c.sonic_slots, dtype=np.int64),
        finger_slots=np.asarray(c.finger_slots, dtype=np.int64),
        sonic_scale=f(c.sonic_scale),
        sonic_default=f(c.sonic_default),
        ref_finger=f(c.ref_finger_target),
        soft_lo=f(c.soft_lower),
        soft_hi=f(c.soft_upper),
        obj_mass=float(c.obj_mass),
        obj_ieff=float(c.obj_inertia_eff),
        obj_com=f(c.obj_com),
        obj_radius=float(c.obj_radius),
        basis=f(c.wrench_basis),
        cone_cos=f(c.cone_cos),
        cone_sin=f(c.cone_sin),
        voc=dict(c.voc),
        obj_cfg=dict(c.objective),
        term=dict(c.termination),
        latent=None if c.latent_table is None else f(c.latent_table),
    )


# ---------------------------------------------------------------------------------------------
# Physical state (world frame). Hand axis order is [right, left] as in the ReconBody observation.


def extract_state(k, body_q, body_qd, joint_q, joint_qd) -> dict:
    pel, obj, wr = k["pelvis"], k["obj"], k["wrist"]
    return dict(
        pp=body_q[:, pel, :3],
        pR=quat_to_mat(body_q[:, pel, 3:7]),
        pv=body_qd[:, pel, :3],
        pw=body_qd[:, pel, 3:6],
        hp=body_q[:, wr, :3],
        hR=quat_to_mat(body_q[:, wr, 3:7]),
        hv=body_qd[:, wr, :3],
        op=body_q[:, obj, :3],
        oR=quat_to_mat(body_q[:, obj, 3:7]),
        ov=body_qd[:, obj, :3],
        ow=body_qd[:, obj, 3:6],
        q=joint_q[:, k["q_ids"]],
        qd=joint_qd[:, k["dof_ids"]],
    )


def palm_frames(k, S):
    """Palm-frame poses [right, left]: body pose composed with the fixed body->palm offset."""
    p = S["hp"] + mv(S["hR"], k["palm_lp"][None])
    R = mm(S["hR"], k["palm_lR"][None])
    return p, R


def _fut(k, frame):
    return jnp.stack([jnp.minimum(frame + o, k["F"] - 1) for o in FUTURE_OFFSETS], axis=-1)  # (B, 3)


# ---------------------------------------------------------------------------------------------
# ReconBody observation (envs/recon_body.py compute_recon_body_observation)


def observation(k, S, frame, target_hist):
    B = S["pp"].shape[0]
    frame = jnp.clip(frame, 0, k["F"] - 1)
    pRt = mT(S["pR"])
    palm_p, palm_R = palm_frames(k, S)
    blocks = []
    blocks.append(mv(pRt[:, None], palm_p - S["pp"][:, None]).reshape(B, 6))
    blocks.append(mat_to_6d(mm(pRt[:, None], palm_R)).reshape(B, 12))
    blocks.append(mv(pRt[:, None], S["hv"]).reshape(B, 6))
    blocks.append(mv(pRt, S["op"] - S["pp"]))
    blocks.append(mat_to_6d(mm(pRt, S["oR"])))
    blocks.append(S["q"] - k["default_q"][None])
    blocks.append(S["qd"])
    fut = _fut(k, frame)
    blocks.append(k["ref_root_p"][fut].reshape(B, 9))
    cur_R = k["ref_root_R"][frame]
    blocks.append(mat_to_6d(mm(k["ref_root_R"][fut], mT(cur_R)[:, None])).reshape(B, 18))
    blocks.append((k["ref_q"][fut] - S["q"][:, None]).reshape(B, 3 * N_JOINTS))
    blocks.append((k["ref_wp"][fut] - palm_p[:, None]).reshape(B, 18))
    blocks.append(mat_to_6d(mm(mT(palm_R)[:, None], k["ref_wR"][fut])).reshape(B, 36))
    oRt = mT(S["oR"])
    lr = jnp.array([1, 0])  # left, right
    hop = mv(oRt[:, None], palm_p[:, lr] - S["op"][:, None])
    hoR = mat_to_6d(mm(oRt[:, None], palm_R[:, lr]))
    blocks.append(jnp.concatenate([hop, hoR], axis=-1).reshape(B, 18))
    blocks.append(jnp.concatenate([mv(oRt, k["ref_op"][frame] - S["op"]), mat_to_6d(mm(oRt, k["ref_oR"][frame]))], -1))
    blocks.append((frame.astype(jnp.float32) / float(k["F"] - 1))[:, None])
    blocks.append(target_hist.reshape(B, 3 * N_JOINTS))
    return jnp.concatenate(blocks, axis=-1)


# ---------------------------------------------------------------------------------------------
# SONIC prior (runtime/actions/sonic.py)


def sonic_current(k, S, last_decoded):
    """compute_sonic_current_observation: [ang vel body(3), q-def(29), qd(29), last decoded(29), gravity_b(3)]."""
    pRt = mT(S["pR"])
    slots = k["sonic_slots"]
    g = mv(pRt, jnp.broadcast_to(jnp.array([0.0, 0.0, -1.0]), S["pp"].shape))
    return jnp.concatenate(
        [mv(pRt, S["pw"]), S["q"][:, slots] - k["sonic_default"][None], S["qd"][:, slots], last_decoded, g], axis=-1
    )


def sonic_decoder_input(latent, hist):
    """pack_sonic_decoder_input; hist (B, 10, 93) ordered oldest -> newest."""
    B = latent.shape[0]
    return jnp.concatenate(
        [
            latent,
            hist[:, :, 0:3].reshape(B, 30),
            hist[:, :, 3:32].reshape(B, 290),
            hist[:, :, 32:61].reshape(B, 290),
            hist[:, :, 61:90].reshape(B, 290),
            hist[:, :, 90:93].reshape(B, 30),
        ],
        axis=-1,
    )


def sonic_decode(dec_params, x, precision=None):
    h = x
    n = len(dec_params) // 2
    for i in range(n):
        h = jnp.matmul(h, dec_params[f"W{i}"], precision=precision) + dec_params[f"b{i}"]
        if i < n - 1:
            h = h * jax.nn.sigmoid(h)
    return h


def processed_targets(k, decoded, action, frame):
    """process_sonic_joint_targets with zero reference offsets (settled worlds)."""
    B = decoded.shape[0]
    body = (decoded + 0.15 * action[:, :SONIC_DIM]) * k["sonic_scale"][None] + k["sonic_default"][None]
    finger = k["ref_finger"][jnp.clip(frame, 0, k["F"] - 1)] + 0.15 * action[:, SONIC_DIM:]
    out = jnp.zeros((B, N_JOINTS))
    out = out.at[:, k["sonic_slots"]].set(body)
    out = out.at[:, k["finger_slots"]].set(finger)
    return out


# ---------------------------------------------------------------------------------------------
# Objective (objectives/recon_body.py compose_recon_body_objective), hand order inside: [right, left]


def objective_terms(k, S, frame, start_frame, action, prev_action, fc, terminated):
    o = k["obj_cfg"]
    F = k["F"]
    frame = jnp.clip(frame, 0, F - 1)
    rate = jnp.sum(jnp.square(action - prev_action), -1)
    l2 = jnp.sum(jnp.square(action), -1)
    q = S["q"]
    limit = jnp.sum(jnp.maximum(k["soft_lo"][None] - q, 0.0) + jnp.maximum(q - k["soft_hi"][None], 0.0), -1)
    pe = jnp.sum(jnp.square(S["pp"] - k["ref_root_p"][frame]), -1)
    pori = rot_angle_sq(mm(mT(S["pR"]), k["ref_root_R"][frame]))
    slots = k["sonic_slots"]
    je = jnp.sum(jnp.square(q[:, slots] - k["ref_q"][frame][:, slots]), -1)
    oe = jnp.sum(jnp.square(S["op"] - k["ref_op"][frame]), -1)
    den = (F - 1 - start_frame).astype(jnp.float32)
    prog = jnp.where(den > 0, jnp.clip((frame - start_frame).astype(jnp.float32) / jnp.maximum(den, 1.0), 0.0, 1.0), 1.0)
    palm_p, palm_R = palm_frames(k, S)
    ppe = jnp.sum(jnp.square(palm_p - k["ref_wp"][frame]), axis=(-1, -2))
    poe = jnp.sum(rot_angle_sq(mm(mT(palm_R), k["ref_wR"][frame])), -1)
    terms = jnp.stack(
        [
            terminated,
            rate,
            l2,
            limit,
            jnp.exp(-pe / o["pelvis_position_std"] ** 2),
            jnp.exp(-pori / o["pelvis_orientation_std"] ** 2),
            jnp.exp(-je / o["body_joint_position_std"] ** 2),
            jnp.exp(-oe / o["object_position_std"] ** 2),
            prog,
            jnp.exp(-ppe / o["palm_position_std"] ** 2),
            jnp.exp(-poe / o["palm_orientation_std"] ** 2),
            fc,
        ],
        axis=-1,
    )
    return terms


def grasp_touch(k, frame, contact_p):
    """Per labelled grasping hand, the expected number of touching links (slots [left x8, right x8]) / 3, capped at 1."""
    lab = k["grasp_lab"][jnp.clip(frame, 0, k["F"] - 1)]  # (B,2) [left,right]
    n = jnp.sum(contact_p.reshape(-1, 2, 8), -1)
    return jnp.sum(jnp.minimum(n / 3.0, 1.0) * lab, -1) / jnp.maximum(jnp.sum(lab, -1), 1.0)


EXTRA_NAMES = ("obj_ori", "obj_fine", "hand_in_obj", "grasp_touch", "obj_vel", "obj_coarse",
               "pelvis_angular_stability", "torso_sideways_lean", "pelvis_velocity",
               "pelvis_angular_velocity", "body_joint_velocity", "finger_grip", "action_residual",
               "obj_ori_coarse", "residual_legs", "residual_arms", "obj_keypoints", "obj_keypoints_fine")


def residual_groups(c):
    """Action dimensions of the legs + waist and of the arms (the first 29 follow sonic_slots)."""
    legs = [i for i, n in enumerate(c.sonic_names) if any(t in n for t in ("hip", "knee", "ankle", "waist"))]
    arms = [i for i in range(len(c.sonic_names)) if i not in legs]
    return np.asarray(legs, np.int64), np.asarray(arms, np.int64)


def finger_grip_table(c, frac, ramp=15, release_lead=0):
    """Commanded finger pose per frame (F, 14). The retargeted reference has no finger motion at all (every target is
    0, i.e. a flat open hand), so holding needs a closing signal: `frac` of the way to the farther limit (thumb base
    rotation centred) on the frames a hand holds the object in the reference, open otherwise."""
    lo, hi = np.asarray(c.soft_lower)[c.finger_slots], np.asarray(c.soft_upper)[c.finger_slots]
    closed = np.array([0.0 if "thumb_0" in n else frac * (lo[i] if abs(lo[i]) > abs(hi[i]) else hi[i])
                       for i, n in enumerate(c.finger_names)], np.float32)
    lab = grasp_labels(c)  # [left, right]
    if release_lead > 0:  # a hand that keeps gripping while the palm leaves drags the object along
        lab = lab * lab[np.minimum(np.arange(len(lab)) + release_lead, len(lab) - 1)]
    closure = np.stack([np.convolve(lab[:, h], np.ones(ramp) / ramp, mode="same") for h in range(2)], 1)
    hand = [0 if n.startswith("left") else 1 for n in c.finger_names]
    return closure[:, hand] * closed


def extra_terms(k, S, frame, contact_p, target, action, hio_std=0.05):
    """Training-only shaping: object orientation (scored, unrewarded upstream), fine object position,
    reference hand pose in the object frame on contact-labelled frames, links touching the object while
    the reference grasps it, object velocity (lift timing), and a wide object position term."""
    frame = jnp.clip(frame, 0, k["F"] - 1)
    ori = rot_angle_sq(mm(mT(S["oR"]), k["ref_oR"][frame]))
    pos = jnp.sum(jnp.square(S["op"] - k["ref_op"][frame]), -1)
    palm_p, _ = palm_frames(k, S)
    live = mv(mT(S["oR"])[:, None], palm_p - S["op"][:, None])  # (B,2,3) [right,left]
    ref = mv(mT(k["ref_oR"][frame])[:, None], k["ref_wp"][frame] - k["ref_op"][frame][:, None])
    lab = k["labels"][frame][:, ::-1]  # labels are [left,right] -> [right,left]
    hio = jnp.exp(-jnp.sum(jnp.square(live - ref), -1) / hio_std**2)
    hio = jnp.sum(hio * lab, -1) / jnp.maximum(jnp.sum(lab, -1), 1.0)
    vel = jnp.exp(-jnp.sum(jnp.square(S["ov"] - k["ref_ov"][frame]), -1) / 0.5**2)
    # Appendix A.6-style approach costs: sideways lean and heading-frame roll/pitch rate.
    walk = (frame < k["first_lift_frame"]).astype(jnp.float32)
    Rh_inv = mT(heading(S))
    omega_h = mv(Rh_inv, S["pw"])
    stable = -walk * jnp.sum(jnp.square(omega_h[:, :2]), -1)
    torso_up_h = mv(Rh_inv, S["pR"][:, :, 2])
    sideways_lean = -walk * jnp.square(torso_up_h[:, 1])
    pelvis_vel = jnp.exp(-jnp.sum(jnp.square(S["pv"] - k["ref_root_v"][frame]), -1) / 0.5**2)
    pelvis_omega = jnp.exp(-jnp.sum(jnp.square(S["pw"] - k["ref_root_w"][frame]), -1) / 3.14**2)
    slots = k["sonic_slots"]
    joint_vel = jnp.exp(-jnp.mean(jnp.square(S["qd"][:, slots] - k["ref_qd"][frame][:, slots]), -1) / 4.0**2)
    # on the command, not the state: its gradient needs no contact data, which only exists once the hand closes
    grip = -jnp.mean(jnp.square(target[:, k["finger_slots"]] - k["grip_q"][frame]), -1)
    # the official action cost (1e-6) lets the policy drift far from SONIC, which never falls on its own
    residual = -jnp.mean(jnp.square(action[:, :SONIC_DIM]), -1) / 100.0
    residual_legs = -jnp.mean(jnp.square(action[:, k["leg_dims"]]), -1) / 100.0
    residual_arms = -jnp.mean(jnp.square(action[:, k["arm_dims"]]), -1) / 100.0
    # mean squared distance of the 6 object keypoints at +-radius along the axes (the +- cross terms cancel)
    keypoints = pos + k["obj_radius"] ** 2 / 3.0 * jnp.sum(jnp.square(S["oR"] - k["ref_oR"][frame]), axis=(-2, -1))
    return jnp.stack(
        [jnp.exp(-ori / 0.4**2), jnp.exp(-pos / 0.05**2), hio, grasp_touch(k, frame, contact_p), vel,
         jnp.exp(-pos / 0.4**2), stable, sideways_lean, pelvis_vel, pelvis_omega, joint_vel, grip, residual,
         jnp.exp(-ori / 1.0**2), residual_legs, residual_arms, jnp.exp(-keypoints / 0.1**2), jnp.exp(-keypoints / 0.03**2)],
        axis=-1,
    )


# ---------------------------------------------------------------------------------------------
# Termination (lifecycle/termination.py evaluate_recon_body_termination)


def tracking_errors(k, S, frame):
    frame = jnp.clip(frame, 0, k["F"] - 1)
    pe = jnp.sqrt(jnp.sum(jnp.square(S["pp"] - k["ref_root_p"][frame]), -1) + 1e-12)
    po = geodesic_angle(S["pR"], k["ref_root_R"][frame])
    palm_p, palm_R = palm_frames(k, S)
    he = jnp.sqrt(jnp.sum(jnp.square(palm_p - k["ref_wp"][frame]), -1) + 1e-12)  # (B,2)
    ho = geodesic_angle(palm_R, k["ref_wR"][frame])
    oe = jnp.sqrt(jnp.sum(jnp.square(S["op"] - k["ref_op"][frame]), -1) + 1e-12)
    oo = geodesic_angle(S["oR"], k["ref_oR"][frame])
    return pe, po, he, ho, oe, oo


def termination_hard(k, S, frame, episode_step):
    t = k["term"]
    pe, po, he, ho, oe, oo = tracking_errors(k, S, frame)
    pel = (pe > t["pelvis_position"]) | (po > t["pelvis_orientation"])
    palm = (jnp.max(he, -1) > t["palm_position"]) | (jnp.max(ho, -1) > t["palm_orientation"])
    obj = (oe > t["object_position"]) | (oo > t["object_orientation"])
    masked = episode_step <= t["freeze"]
    return pel | ((palm | obj) & ~masked)


def termination_soft(k, S, frame, episode_step, tau_frac=0.1, free=None):
    """`free` (B,) = 1 ignores the palm and object limits: rollouts that start past them (evaluator-style frame-0 episodes)."""
    t = k["term"]
    pe, po, he, ho, oe, oo = tracking_errors(k, S, frame)
    sig = lambda e, thr: jax.nn.sigmoid((e - thr) / (tau_frac * thr))  # noqa: E731
    live = (1 - sig(pe, t["pelvis_position"])) * (1 - sig(po, t["pelvis_orientation"]))
    gate = (episode_step > t["freeze"]).astype(jnp.float32)
    if free is not None:
        gate = gate * (1.0 - free)
    for e, thr in ((he, t["palm_position"]), (ho, t["palm_orientation"])):
        live = live * jnp.prod(1 - gate[:, None] * sig(e, thr), -1)
    for e, thr in ((oe, t["object_position"]), (oo, t["object_orientation"])):
        live = live * (1 - gate * sig(e, thr))
    return 1.0 - live


# ---------------------------------------------------------------------------------------------
# Force closure (objectives/wrench.py batched_wrench_support + reduce_recon_body_force_closure)


def _cone_rays(k, n):
    """Friction-cone rays (edges + normal) for unit normals n (..., 3) -> (..., E+1, 3)."""
    mu = k["obj_cfg"]["friction"]
    sign = jnp.where(n[..., 2] >= 0.0, 1.0, -1.0)
    a = -1.0 / (sign + n[..., 2])
    bb = n[..., 0] * n[..., 1] * a
    t1 = jnp.stack([1.0 + sign * n[..., 0] ** 2 * a, sign * bb, -sign * n[..., 0]], -1)
    t2 = jnp.stack([bb, sign + n[..., 1] ** 2 * a, -n[..., 1]], -1)
    nrm = lambda v: v / jnp.sqrt(jnp.sum(v * v, -1, keepdims=True) + 1e-12)  # noqa: E731
    t1, t2 = nrm(t1), nrm(t2)
    edges = nrm(
        n[..., None, :]
        + mu * (k["cone_cos"][:, None] * t1[..., None, :] + k["cone_sin"][:, None] * t2[..., None, :])
    )
    return jnp.concatenate([edges, n[..., None, :]], axis=-2)


def force_closure(k, frame, active, pos, direction, sharp=None):
    """active (B,16) in [0,1]; pos/direction (B,16,3) object frame; slots ordered [left x8, right x8].

    sharp=None reproduces the hard env value (for validation); otherwise a sigmoid temperature.
    """
    B = active.shape[0]
    n = direction / jnp.sqrt(jnp.sum(direction * direction, -1, keepdims=True) + 1e-12)
    rays = _cone_rays(k, n)  # (B,16,9,3)
    torque = jnp.cross(pos[:, :, None, :], rays) / k["obj_radius"]
    w = jnp.concatenate([rays, torque], -1)  # (B,16,9,6)
    vals = jnp.einsum("bcef,nf->bcen", w, k["basis"])  # (B,16,9,512)
    valid = active * (jnp.sum(direction * direction, -1) > 1e-6)
    vals = vals * valid[:, :, None, None]
    vals = vals.reshape(B, 2, 8 * 9, -1)
    support = jnp.maximum(jnp.max(vals, axis=2), 0.0)  # (B,2,512)
    thr = k["obj_cfg"]["force_closure_min_support"]
    if sharp is None:
        cov = jnp.mean((support > thr).astype(jnp.float32), -1)
    else:
        cov = jnp.mean(jax.nn.sigmoid((support - thr) / sharp), -1)
    lab = k["labels"][jnp.clip(frame, 0, k["F"] - 1)]  # (B,2) [left,right] = slot hand order
    return jnp.sum(cov * lab, -1) / jnp.maximum(jnp.sum(lab, -1), 1.0)


# ---------------------------------------------------------------------------------------------
# World-model state encoding in the pelvis heading frame


def heading(S):
    return rot_z(yaw_of(S["pR"]))


def wm_state_features(S):
    Rh = heading(S)
    Rht = mT(Rh)
    return jnp.concatenate(
        [
            S["pp"][:, 2:3],
            mat_to_6d(mm(Rht, S["pR"])),
            mv(Rht, S["pv"]),
            mv(Rht, S["pw"]),
            mv(Rht[:, None], S["hp"] - S["pp"][:, None]).reshape(-1, 6),
            mat_to_6d(mm(Rht[:, None], S["hR"])).reshape(-1, 12),
            mv(Rht[:, None], S["hv"]).reshape(-1, 6),
            mv(Rht, S["op"] - S["pp"]),
            S["op"][:, 2:3],
            mat_to_6d(mm(Rht, S["oR"])),
            mv(Rht, S["ov"]),
            mv(Rht, S["ow"]),
            S["q"],
            S["qd"],
        ],
        axis=-1,
    )


STATE_FEAT_DIM = 1 + 6 + 3 + 3 + 6 + 12 + 6 + 3 + 1 + 6 + 3 + 3 + 43 + 43  # 139


def _local(S, Rht, origin):
    """Next-state quantities expressed in a fixed (previous-step) heading frame."""
    return dict(
        pp=mv(Rht, S["pp"] - origin),
        pR=mat_to_6d(mm(Rht, S["pR"])),
        pv=mv(Rht, S["pv"]),
        pw=mv(Rht, S["pw"]),
        hp=mv(Rht[:, None], S["hp"] - origin[:, None]).reshape(-1, 6),
        hR=mat_to_6d(mm(Rht[:, None], S["hR"])).reshape(-1, 12),
        hv=mv(Rht[:, None], S["hv"]).reshape(-1, 6),
        op=mv(Rht, S["op"] - origin),
        oR=mat_to_6d(mm(Rht, S["oR"])),
        ov=mv(Rht, S["ov"]),
        ow=mv(Rht, S["ow"]),
        q=S["q"],
        qd=S["qd"],
    )


DELTA_KEYS = ("pp", "pR", "pv", "pw", "hp", "hR", "hv", "op", "oR", "ov", "ow", "q", "qd")
DELTA_DIMS = (3, 6, 3, 3, 6, 12, 6, 3, 6, 3, 3, 43, 43)
DELTA_DIM = sum(DELTA_DIMS)  # 140


def delta_weights(object_weight):
    """Per-dimension world-model loss weights (mean 1) that favour the object part of the state delta."""
    w = np.ones(DELTA_DIM, np.float32)
    i = 0
    for key, d in zip(DELTA_KEYS, DELTA_DIMS):
        if key in ("op", "oR", "ov", "ow"):
            w[i : i + d] = object_weight
        i += d
    return jnp.asarray(w / w.mean())


def encode_delta(S, S_next):
    """Target residual: next state in the current heading frame minus current state in that frame."""
    Rh = heading(S)
    Rht = mT(Rh)
    a = _local(S, Rht, S["pp"])
    b = _local(S_next, Rht, S["pp"])
    return jnp.concatenate([b[key] - a[key] for key in DELTA_KEYS], -1)


def decode_delta(S, delta):
    Rh = heading(S)
    Rht = mT(Rh)
    a = _local(S, Rht, S["pp"])
    parts = {}
    i = 0
    for key, d in zip(DELTA_KEYS, DELTA_DIMS):
        parts[key] = a[key] + delta[:, i : i + d]
        i += d
    B = delta.shape[0]
    origin = S["pp"]
    return dict(
        pp=origin + mv(Rh, parts["pp"]),
        pR=mm(Rh, six_d_to_mat(parts["pR"])),
        pv=mv(Rh, parts["pv"]),
        pw=mv(Rh, parts["pw"]),
        hp=origin[:, None] + mv(Rh[:, None], parts["hp"].reshape(B, 2, 3)),
        hR=mm(Rh[:, None], six_d_to_mat(parts["hR"].reshape(B, 2, 6))),
        hv=mv(Rh[:, None], parts["hv"].reshape(B, 2, 3)),
        op=origin + mv(Rh, parts["op"]),
        oR=mm(Rh, six_d_to_mat(parts["oR"])),
        ov=mv(Rh, parts["ov"]),
        ow=mv(Rh, parts["ow"]),
        q=parts["q"],
        qd=parts["qd"],
    )


def voc_features(k, S, frame, voc_scale):
    """Analytic VOC wrench at the start of the step (runtime/object_control.py), in heading frame,
    plus the target pose error. VOC is a known external input, so the world model is conditioned on it."""
    v = k["voc"]
    frame = jnp.clip(frame, 0, k["F"] - 1)
    Rht = mT(heading(S))
    tp, tR = k["ref_op"][frame], k["ref_oR"][frame]
    com_w = mv(S["oR"], jnp.broadcast_to(k["obj_com"], S["op"].shape))
    lin = S["ov"] - jnp.cross(S["ow"], com_w)
    force = v["k_lin"] * (tp - S["op"]) - v["d_lin"] * lin
    rv = log_map(mm(tR, mT(S["oR"])))
    torque = v["k_ang"] * rv - v["d_ang"] * S["ow"]
    if v["mass_normalized"]:
        force, torque = k["obj_mass"] * force, k["obj_ieff"] * torque
    force = force + jnp.array([0.0, 0.0, 1.0]) * v["gravity"] * k["obj_mass"]
    s = voc_scale[:, None]
    force = jnp.clip(s * force, -v["max_force"], v["max_force"])
    torque = jnp.clip(s * torque, -v["max_torque"], v["max_torque"])
    return jnp.concatenate([voc_scale[:, None], mv(Rht, force) / 60.0, mv(Rht, torque) / 60.0, mv(Rht, tp - S["op"]), mv(Rht, rv)], -1)


VOC_FEAT_DIM = 13


def hand_object_features(k, S):
    """Palm position in the object frame and wrist velocity relative to the object: contact and slip depend on these
    directly, and the MLP would otherwise have to rebuild them from two pelvis-relative poses."""
    palm_p, _ = palm_frames(k, S)
    rel = mv(mT(S["oR"])[:, None], palm_p - S["op"][:, None]).reshape(-1, 6)
    vel = mv(mT(heading(S))[:, None], S["hv"] - S["ov"][:, None]).reshape(-1, 6)
    return jnp.concatenate([rel, vel], -1)


HAND_OBJECT_DIM = 12


def wm_inputs(k, S, target, prev_target, lag_onehot, voc_scale, frame):
    return jnp.concatenate(
        [wm_state_features(S), hand_object_features(k, S), target, prev_target, target - S["q"], lag_onehot,
         voc_features(k, S, frame, voc_scale)], -1
    )


WM_IN_DIM = STATE_FEAT_DIM + HAND_OBJECT_DIM + 43 * 3 + 15 + VOC_FEAT_DIM
N_SLOTS = 16
WM_OUT_DIM = DELTA_DIM + N_SLOTS * 7 + 1  # delta + contact (logit, pos3, dir3) + force-closure logit


def lag_onehot(lags):
    return jax.nn.one_hot(lags, 3).reshape(lags.shape[0], 15)


def critic_features(k, S, frame, episode_step, voc_scale, lag1h):
    F = k["F"]
    return jnp.concatenate(
        [
            wm_state_features(S),
            (jnp.clip(frame, 0, F - 1).astype(jnp.float32) / (F - 1))[:, None],
            jnp.clip(episode_step.astype(jnp.float32) / 51.0, 0.0, 1.0)[:, None],
            voc_scale[:, None],
            lag1h,
            *[mv(mT(heading(S)), k["ref_op"][jnp.clip(frame, 0, F - 1)] - S["op"])],
        ],
        -1,
    )


CRITIC_IN_DIM = STATE_FEAT_DIM + 3 + 15 + 3
