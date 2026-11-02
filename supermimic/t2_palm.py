"""Make the palm targets of a retargeted Tier 2 motion follow the object rigidly during holds, using Tier 2 alone (Tier 1 is never read, PLAN.md 13.22.8).

The human carries its own walk, so the palm drifts on the object while the hand does not actually move on it, and the policy is asked to hold the object with a palm that
keeps turning: it grasps at a wrong angle (ep 3 keeps the mug 40-50 deg off from the first lift on). A hold is rigid, and the walk starts near zero, so the grasp is
estimated from the first frames of a hold, and carried over to later holds on the same spot. During the hold the palm target is then the anchored object pose times that
grasp. Holds in which the hand really slides on the object keep their targets.
usage: python -m supermimic.t2_palm <tier_2 dir> <motion root in> <motion root out> [sequence substring ...]
"""

import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

from supermimic.t2_grasp_prior import runs, set_row

MIN_HOLD = 30  # frames of a hold to trust it
SKIP = 8  # frames at the start of a hold while the fingers still close
WINDOW = 30  # frames estimating the grasp
SAME_SPOT_M = 0.05  # a later hold this close to the last grasp is the same grasp
RIGID_RMS_M = 0.05  # palm spread on the object above this: the hand slides, this is not a rigid hold
RAMP = 15  # frames to fade the correction in and out


def rigid_palms(ee, labels, obj_p, obj_r):
    out = ee.copy()
    changed, frames = [], 0
    for h in range(2):
        palm_p, palm_r = ee[:, h, :3], R.from_quat(ee[:, h, [4, 5, 6, 3]])
        rel_p = obj_r.inv().apply(palm_p - obj_p)
        rel_r = obj_r.inv() * palm_r
        grasp = None
        for a, b in runs(labels[h]):
            if b - a < MIN_HOLD:
                continue
            head = slice(a + SKIP, min(a + SKIP + WINDOW, b))
            p, r = rel_p[head].mean(0), rel_r[head].mean()
            if grasp is not None and np.linalg.norm(p - grasp[0]) < SAME_SPOT_M:
                p, r = grasp
            grasp = (p, r)
            if np.sqrt(((rel_p[a:b] - p) ** 2).sum(-1).mean()) > RIGID_RMS_M:
                continue
            weight = np.minimum(np.minimum(np.arange(b - a) + 1, np.arange(b - a)[::-1] + 1) / RAMP, 1.0)
            target_p = obj_p[a:b] + obj_r[a:b].apply(p)
            target_r = obj_r[a:b] * r
            new_r = palm_r[a:b] * R.from_rotvec((palm_r[a:b].inv() * target_r).as_rotvec() * weight[:, None])
            out[a:b, h, :3] = palm_p[a:b] + weight[:, None] * (target_p - palm_p[a:b])
            out[a:b, h, 3:7] = new_r.as_quat()[:, [3, 0, 1, 2]]
            changed.extend(np.degrees((palm_r[a:b].inv() * new_r).magnitude()).tolist())
            frames += b - a
    return out, frames, float(np.median(changed)) if changed else 0.0, float(np.max(changed)) if changed else 0.0


def process(path_in, path_out):
    table = pq.read_table(path_in)
    row = table.to_pydict()
    labels = np.asarray(row["hand_contact_active"][0], float) > 0.5
    obj_p = np.asarray(row["object_body_position"][0], float)[:, 0]
    obj_r = R.from_quat(np.asarray(row["object_body_wxyz"][0], float)[:, 0, [1, 2, 3, 0]])
    ee, frames, median, top = rigid_palms(np.asarray(row["ee_pose_w"][0], float), labels, obj_p, obj_r)
    pq.write_table(set_row(table, "ee_pose_w", ee), path_out)
    return frames, median, top


def main(tier, src, dst, wanted):
    os.makedirs(f"{dst}/whole_body/soma", exist_ok=True)
    if not os.path.lexists(f"{dst}/whole_body/reconstructed_stage"):
        os.symlink(os.path.abspath(f"{src}/whole_body/reconstructed_stage"), f"{dst}/whole_body/reconstructed_stage")
    with open(f"{tier}/meta/episodes_metadata.jsonl") as f:
        meta = sorted((json.loads(line) for line in f), key=lambda d: d["episode_index"])
    for d in meta:
        seq_dir = f"{src}/whole_body/soma/sequence_id={d['sequence_id']}"
        if (wanted and not any(w in d["sequence_id"] for w in wanted)) or not os.path.exists(f"{seq_dir}/robot_name=g1/data.parquet"):
            continue
        out_dir = f"{dst}/whole_body/soma/sequence_id={d['sequence_id']}"
        os.makedirs(f"{out_dir}/robot_name=g1", exist_ok=True)
        for item in os.listdir(seq_dir):
            if item != "robot_name=g1" and not os.path.lexists(f"{out_dir}/{item}"):
                os.symlink(os.path.abspath(f"{seq_dir}/{item}"), f"{out_dir}/{item}")
        frames, median, top = process(f"{seq_dir}/robot_name=g1/data.parquet", f"{out_dir}/robot_name=g1/data.parquet")
        print(f"ep{d['episode_index']} {d['object'][:18]}: {frames} palm frames made rigid, palm target turned median {median:.0f} max {top:.0f} deg", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
