"""Physical priors on a retargeted Tier 2 motion, using Tier 2 alone (Tier 1 is never read).

Two kinds of corruption can be reduced without any clean data:
- Contact labels flicker (median 8 switches per hand, up to 32), and they gate the force closure reward: short gaps are closed and short runs removed.
- A rigid grasp keeps the object fixed in the palm, but hand and object drift independently. While a hand holds, the object is pulled halfway towards the
  pose predicted from the palm and the mean grasp of that interval, so that the reference does not ask for an impossible relative motion.
usage: python -m supermimic.t2_grasp_prior <motion root in> <motion root out> [sequence substring ...]
"""

import glob
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

MAX_GAP = 15  # frames of a contact gap to close
MIN_RUN = 10  # frames of a contact run to keep
MIN_HOLD = 30  # frames of a hold to trust its mean grasp
BLEND = 0.5  # how far the object moves towards the palm prediction
RAMP = 15  # frames to fade the correction in and out
RIGID_RMS_M = 0.03  # object spread in the palm frame above this is not a rigid grasp (the input noise alone is 1-2 cm)


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2]))


def clean_labels(lab):
    lab = lab.copy()
    on = runs(lab)
    for (_, b), (c, _) in zip(on, on[1:]):
        if c - b < MAX_GAP:
            lab[b:c] = True
    for a, b in runs(lab):
        if b - a < MIN_RUN:
            lab[a:b] = False
    return lab


def wxyz_to_rot(q):
    return R.from_quat(q[..., [1, 2, 3, 0]])


def fuse(obj_p, obj_q, ee, labels):
    """Object position and wxyz quaternion pulled towards the palm predictions on the hold intervals."""
    T = len(obj_p)
    sum_p, sum_q, weight = np.zeros((T, 3)), np.zeros((T, 4)), np.zeros(T)
    obj_r = wxyz_to_rot(obj_q)
    for h in range(2):
        palm_p, palm_r = ee[:, h, :3], wxyz_to_rot(ee[:, h, 3:7])
        for a, b in runs(labels[h]):
            if b - a < MIN_HOLD:
                continue
            rel_p = palm_r[a:b].inv().apply(obj_p[a:b] - palm_p[a:b])
            spread = np.sqrt(((rel_p - np.median(rel_p, axis=0)) ** 2).sum(-1).mean())
            if spread > RIGID_RMS_M:  # the hand slides on the object or changes grip: not a rigid grasp, so no prediction from the palm
                continue
            rel_r = (palm_r[a:b].inv() * obj_r[a:b]).mean()
            pred_p = palm_p[a:b] + palm_r[a:b].apply(np.median(rel_p, axis=0))
            pred_q = (palm_r[a:b] * rel_r).as_quat()[:, [3, 0, 1, 2]]
            pred_q = pred_q * np.sign((pred_q * obj_q[a:b]).sum(-1, keepdims=True) + 1e-9)
            ramp = np.minimum(np.minimum(np.arange(b - a) + 1, np.arange(b - a)[::-1] + 1) / RAMP, 1.0)
            sum_p[a:b] += ramp[:, None] * pred_p
            sum_q[a:b] += ramp[:, None] * pred_q
            weight[a:b] += ramp
    held = weight > 0
    out_p, out_q = obj_p.copy(), obj_q.copy()
    pull = BLEND * np.minimum(weight, 1.0)[held, None]
    out_p[held] = (1 - pull) * obj_p[held] + pull * sum_p[held] / weight[held, None]
    out_q[held] = (1 - pull) * obj_q[held] + pull * sum_q[held] / weight[held, None]
    return out_p, out_q / np.linalg.norm(out_q, axis=-1, keepdims=True)


def set_row(table, name, array):
    field = table.schema.field(name)
    return table.set_column(table.schema.get_field_index(name), field, pa.array([array.tolist()], type=field.type))


def process(path_in, path_out):
    table = pq.read_table(path_in)
    row = table.to_pydict()
    lab = np.asarray(row["hand_contact_active"][0], float) > 0.5
    labels = np.stack([clean_labels(lab[h]) for h in range(2)])
    obj_p = np.asarray(row["object_body_position"][0], float)[:, 0]
    obj_q = np.asarray(row["object_body_wxyz"][0], float)[:, 0]
    ee = np.asarray(row["ee_pose_w"][0], float)
    new_p, new_q = fuse(obj_p, obj_q, ee, labels)
    dtype = np.asarray(row["hand_contact_active"][0]).dtype
    table = set_row(table, "hand_contact_active", labels.astype(dtype))
    table = set_row(table, "object_body_position", new_p[:, None])
    table = set_row(table, "object_body_wxyz", new_q[:, None])
    table = set_row(table, "object_root_position", new_p)
    table = set_row(table, "object_root_axis_angle", wxyz_to_rot(new_q).as_rotvec())
    pq.write_table(table, path_out)
    switches = [int(np.abs(np.diff(lab[h].astype(int))).sum()) for h in range(2)], [int(np.abs(np.diff(labels[h].astype(int))).sum()) for h in range(2)]
    return switches, float(100 * np.linalg.norm(new_p - obj_p, axis=1).mean()), float(100 * np.linalg.norm(new_p - obj_p, axis=1).max())


def main(src, dst, wanted):
    os.makedirs(f"{dst}/whole_body/soma", exist_ok=True)
    for name in ("reconstructed_stage",):
        if not os.path.lexists(f"{dst}/whole_body/{name}"):
            os.symlink(os.path.abspath(f"{src}/whole_body/{name}"), f"{dst}/whole_body/{name}")
    for path in sorted(glob.glob(f"{src}/whole_body/soma/sequence_id=*/robot_name=g1/data.parquet")):
        seq_dir = os.path.dirname(os.path.dirname(path))
        seq = os.path.basename(seq_dir)
        if wanted and not any(w in seq for w in wanted):
            continue
        out_dir = f"{dst}/whole_body/soma/{seq}"
        os.makedirs(f"{out_dir}/robot_name=g1", exist_ok=True)
        for item in os.listdir(seq_dir):
            if item != "robot_name=g1" and not os.path.lexists(f"{out_dir}/{item}"):
                os.symlink(os.path.abspath(f"{seq_dir}/{item}"), f"{out_dir}/{item}")
        sw, mean, top = process(path, f"{out_dir}/robot_name=g1/data.parquet")
        print(f"{seq[12:60]}: label switches {sw[0]} -> {sw[1]}; object moved mean {mean:.2f} max {top:.2f} cm", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
