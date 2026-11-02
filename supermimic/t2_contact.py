"""Repair the contact labels of a retargeted Tier 2 motion from geometry, using Tier 2 alone (Tier 1 is never read, PLAN.md 13.22.8).

The retargeter calls a hand in contact only within 2 cm of an object vertex. With noise on both the human and the object that loses most holds (ep 1 keeps 66 frames
of its left hand, ep 19 81 and 122), and those labels gate the force closure reward, the largest term of the objective. Physics gives a better rule: an object that moves
is being moved, so a hand close to it then counts as holding it, and a hand that stays close after a labelled frame keeps holding.
usage: python -m supermimic.t2_contact <tier_2 dir> <motion root in> <motion root out> [sequence substring ...]
"""

import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from supermimic.t2_anchor import still_segments
from supermimic.t2_grasp_prior import clean_labels, runs, set_row

NEAR_M = 0.10  # palm to surface: the palm link sits a few cm from what the fingers hold, plus the noise of the human


def repair(labels, distance, moving):
    held = clean_labels(labels)
    near = distance < NEAR_M
    out = held | (near & moving)
    for a, b in runs(near):
        if held[a:b].any():
            out[a:b] = True
    return clean_labels(out)


def process(path_in, path_out, surface):
    table = pq.read_table(path_in)
    row = table.to_pydict()
    labels = np.asarray(row["hand_contact_active"][0], float) > 0.5
    obj_p = np.asarray(row["object_body_position"][0], float)[:, 0]
    obj_r = R.from_quat(np.asarray(row["object_body_wxyz"][0], float)[:, 0, [1, 2, 3, 0]])
    ee = np.asarray(row["ee_pose_w"][0], float)
    moving = np.ones(len(obj_p), bool)
    for a, b in still_segments(obj_p, obj_r):
        moving[a:b] = False
    fixed = np.stack([repair(labels[h], surface.query(obj_r.inv().apply(ee[:, h, :3] - obj_p))[0], moving) for h in range(2)])
    dtype = np.asarray(row["hand_contact_active"][0]).dtype
    pq.write_table(set_row(table, "hand_contact_active", fixed.astype(dtype)), path_out)
    covered = (labels.any(0) & moving).sum() / max(moving.sum(), 1), (fixed.any(0) & moving).sum() / max(moving.sum(), 1)
    return labels.sum(1).tolist(), fixed.sum(1).tolist(), covered


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
        mesh = trimesh.load(f"{tier}/{d['mesh']}", force="mesh")
        surface = cKDTree(trimesh.sample.sample_surface(mesh, 20000, seed=0)[0])
        before, after, covered = process(f"{seq_dir}/robot_name=g1/data.parquet", f"{out_dir}/robot_name=g1/data.parquet", surface)
        print(f"ep{d['episode_index']} {d['object'][:18]}: labelled frames L,R {before} -> {after}; moving frames with a hand on {covered[0]:.2f} -> {covered[1]:.2f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
