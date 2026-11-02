"""Remove the Tier 2 random walk from the object pose with physical anchors, using Tier 2 alone (Tier 1 is never read, PLAN.md 13.22.8).

What Tier 2 itself shows (t2_walk.py): frame 0 is nearly clean (tilt from a stable mesh face 1 deg, clearance 0 cm) and the error then grows like a walk of about
1 deg per sqrt(frame), up to about 0.5 rad and 10 cm. A slow walk passes any low-pass filter, but it can be observed wherever the true pose is known:
- While nobody touches a still object its true pose is constant, so all the wobble on that segment is the walk, on all six degrees of freedom.
- When it comes to rest it lies on a stable face of its mesh on the ground or a support, which fixes its tilt and height, but neither its yaw nor x, y. This also
  holds while a hand still touches it, if it stands still on a surface.
Between two anchors the mean of a walk given both ends is the straight line, so the correction is interpolated (a Brownian bridge). Yaw and x, y of a new rest
carry the correction of the last anchor, so they only keep the walk of the motion in between. The correction acts on the world side of the rotation: on the object side
the tilt predicted for the next rest is much worse (ep 3 26 instead of 3 deg, ep 5 46 instead of 6). The human gets the low-pass of t2_filter.py.
usage: python -m supermimic.t2_anchor <tier_2 dir> <contact motion root> <output dir> [episode ...]    other dataset folders are symlinked
"""

import glob
import json
import os
import re
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from supermimic.t2_filter import BLEND_FRAMES, FPS, OBJECT_CUTOFF_HZ, STILL_RADIUS_M, column, fill_dropouts, filter_human, lowpass
from supermimic.t2_grasp_prior import clean_labels

MIN_FREE = 30  # frames nobody touches a still segment, to call it a rest and not a still hold
TRIM = 5  # frames at both ends of a rest that may still move
WINDOW = 15  # frames averaged at the end of a rest and at the start of the next one
MAX_TILT_DEG = 35.0  # the walk saturates near 29 deg: a farther stable face is not where the object lies
MAX_DROP_M = 0.15  # the walk saturates near 10 cm: a farther surface is not what the object lies on
STILL_DEG = 20.0  # turned over +-0.5 s above this: it is being rotated in place (the walk alone gives about 6 deg, 15 at its fastest)
STILL_MIN_FRAMES = 20
ON_SURFACE_M = 0.04  # a still object that a hand touches counts as lying there only this close to a surface


def still_segments(pos, rot):
    """Segments that neither move nor turn over +-0.5 s."""
    half = int(FPS / 2)
    n = len(pos)
    ahead, behind = np.minimum(np.arange(n) + half, n - 1), np.maximum(np.arange(n) - half, 0)
    still = (np.linalg.norm(pos[ahead] - pos[behind], axis=1) < STILL_RADIUS_M) & (np.degrees((rot[ahead] * rot[behind].inv()).magnitude()) < STILL_DEG)
    edges = np.flatnonzero(np.diff(np.r_[0, still.astype(int), 0]))
    return [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= STILL_MIN_FRAMES]


def rot_of(quat_wxyz):
    return R.from_quat(quat_wxyz[..., [1, 2, 3, 0]])


def wxyz_of(rot):
    return rot.as_quat()[..., [3, 0, 1, 2]]


def stable_ups(hull):
    """Up axes in the object frame of the faces the mesh can rest on."""
    transforms, _ = trimesh.poses.compute_stable_poses(hull, n_samples=1, threshold=0.01)
    return np.array([t[:3, :3].T @ [0.0, 0.0, 1.0] for t in transforms])


def support_tops(path):
    """Heights of the support surfaces above the ground, in the G1 frame of the USD stage, whose ground is the fitted plane."""
    tops = []
    with open(path) as f:
        for block in re.split(r"\bdef (?:Cube|Cylinder) ", f.read())[1:]:
            z = re.search(r"xformOp:translate = \(([^)]*)\)", block)
            half = re.search(r"double height = ([\d.e-]+)", block)
            half = float(half.group(1)) / 2 if half else None
            if half is None:
                scale = re.search(r"xformOp:scale = \(([^)]*)\)", block)
                size = re.search(r"double size = ([\d.e-]+)", block)
                half = float(scale.group(1).split(",")[2]) * (float(size.group(1)) if size else 2.0) / 2 if scale else None
            if z and half is not None:
                tops.append(float(z.group(1).split(",")[2]) + half)
    return tops


def align(v, n):
    """The smallest rotation taking the unit vector v to the unit vector n."""
    axis = np.cross(v, n)
    size = np.linalg.norm(axis)
    if size < 1e-9:
        return R.identity()
    return R.from_rotvec(axis / size * np.arctan2(size, float(v @ n)))


def settle(rot, pos, ups, hull, n, c, heights):
    """The predicted pose of a resting object moved onto its nearest stable face and nearest surface. Returns it and the tilt it had."""
    world_ups = rot.apply(ups)
    angles = np.degrees(np.arccos(np.clip(world_ups @ n, -1.0, 1.0)))
    k = int(angles.argmin())
    if angles[k] < MAX_TILT_DEG:
        rot = align(world_ups[k], n) * rot
    clearance = (rot.apply(hull.vertices) @ n).min() + pos @ n + c
    drop = np.array(heights) - clearance
    if np.abs(drop).min() < MAX_DROP_M:
        pos = pos + drop[np.abs(drop).argmin()] * n
    return rot, pos, float(angles[k]), float(np.abs(drop).min())


def anchor_object(pose, visible, touch, hull, ups, n, c, tops):
    """pose: [x, y, z, qw, qx, qy, qz] per frame. Returns the corrected pose and a line per rest."""
    pos, quat = fill_dropouts(pose[:, :3], pose[:, 3:], visible)
    rot = rot_of(quat)
    T = len(pos)
    if pos[0] @ n + c < 0:
        n, c = -n, -c
    ups = np.vstack([ups, rot[0].inv().apply(n)])  # the pose it starts in can be a stable lean on something
    start_clearance = (rot[0].apply(hull.vertices) @ n).min() + pos[0] @ n + c
    heights = [0.0, start_clearance, *tops]
    rests = still_segments(pos, rot)
    if not rests or rests[0][0] > 0:
        rests.insert(0, (0, 1))
    frames, corr_r, corr_d = [], [], []
    last_r, last_d = R.identity(), np.zeros(3)
    lines = []
    still_pose = {}
    for index, (a, b) in enumerate(rests):
        free = (~touch[a:b]).sum() >= MIN_FREE
        first = a + TRIM if b - a > 2 * TRIM else a
        last = b - TRIM if b - a > 2 * TRIM else b
        if index == 0:
            head = range(0, min(5, b))
            rest_r, rest_p = rot[head[0] : head[-1] + 1].mean(), pos[head[0] : head[-1] + 1].mean(0)
            tilt, drop = 0.0, 0.0
        else:
            head = slice(first, min(first + WINDOW, last))
            predicted_r = (last_r * rot[head]).mean()
            predicted_p = (pos[head] + last_d).mean(0)
            raw_r = rot[head].mean()
            raw_tilt = np.degrees(np.arccos(np.clip(raw_r.apply(ups) @ n, -1.0, 1.0))).min()
            rest_r, rest_p, tilt, drop = settle(predicted_r, predicted_p, ups, hull, n, c, heights)
            if not free and (drop > ON_SURFACE_M or tilt > MAX_TILT_DEG):  # a still hold in the air tells nothing
                continue
            lines.append(f"{'' if free else 'touched '}[{a},{b}) tilt raw {raw_tilt:.0f} -> predicted {tilt:.0f} deg, surface off {100 * drop:.1f} cm")
        still_pose[(a, b)] = (rest_r, rest_p)
        frames.extend([first, last - 1] if last - 1 > first else [first])
        tail = slice(max(first, last - WINDOW), last)
        last_r = rest_r * rot[tail].mean().inv()
        last_d = rest_p - pos[tail].mean(0)
        begin_r = rest_r * rot[slice(first, min(first + WINDOW, last))].mean().inv()
        begin_d = rest_p - pos[slice(first, min(first + WINDOW, last))].mean(0)
        if last - 1 > first:
            corr_r.extend([begin_r, last_r])
            corr_d.extend([begin_d, last_d])
        else:
            corr_r.append(last_r)
            corr_d.append(last_d)
    knots = np.array(frames)
    if len(knots) == 1:
        est_r = corr_r[0] * rot
        est_p = pos + corr_d[0]
    else:
        clamped = np.clip(np.arange(T), knots[0], knots[-1])
        est_r = Slerp(knots, R.concatenate(corr_r))(clamped) * rot
        est_p = pos + np.stack([np.interp(np.arange(T), knots, np.array(corr_d)[:, i]) for i in range(3)], -1)
    # zero-phase low-pass of the motion; rests stay exactly constant, with the same ramp as t2_filter.py
    q = wxyz_of(est_r)
    for i in range(1, T):
        if q[i] @ q[i - 1] < 0:
            q[i] = -q[i]
    smooth_p, smooth_q = lowpass(est_p, OBJECT_CUTOFF_HZ), lowpass(q, OBJECT_CUTOFF_HZ)
    held_p, held_q, weight = est_p.copy(), q.copy(), np.zeros(T)
    for (a, b), (rest_r, rest_p) in still_pose.items():
        if b - a < 2:
            continue
        held_p[a:b] = rest_p
        rest_q = wxyz_of(rest_r)
        held_q[a:b] = rest_q * np.sign((q[a:b] @ rest_q)[:, None] + 1e-9)
        weight[a:b] = np.minimum(np.minimum(np.arange(b - a) + 1, np.arange(b - a)[::-1] + 1) / BLEND_FRAMES, 1.0)
    w = weight[:, None]
    out_p = w * held_p + (1 - w) * smooth_p
    out_q = w * held_q + (1 - w) * smooth_q
    return np.concatenate([out_p, out_q / np.linalg.norm(out_q, axis=-1, keepdims=True)], -1), lines


def main(src, contact_root, dst, episodes):
    os.makedirs(f"{dst}/data/chunk-000", exist_ok=True)
    for name in os.listdir(src):
        if name != "data" and not os.path.exists(f"{dst}/{name}"):
            os.symlink(os.path.abspath(f"{src}/{name}"), f"{dst}/{name}")
    meta = {}
    with open(f"{src}/meta/episodes_metadata.jsonl") as f:
        for line in f:
            d = json.loads(line)
            meta[d["episode_index"]] = d
    for path in sorted(glob.glob(f"{src}/data/chunk-000/episode_*.parquet")):
        ep = int(os.path.basename(path)[8:14])
        if episodes and ep not in episodes:
            continue
        d = meta[ep]
        hull = trimesh.load(f"{src}/{d['mesh']}", force="mesh").convex_hull
        with open(f"{src}/{d['ground_plane']}") as f:
            plane = np.array(json.load(f)["plane"])
        contact = pq.read_table(f"{contact_root}/whole_body/soma/sequence_id={d['sequence_id']}/robot_name=g1/data.parquet", columns=["hand_contact_active"]).to_pydict()
        labels = np.asarray(contact["hand_contact_active"][0], float) > 0.5
        touch = clean_labels(labels[0]) | clean_labels(labels[1])
        t = pq.read_table(path)
        pose = np.asarray(t["observation.object.pose"].to_pylist(), np.float64)
        visible = np.asarray(t["observation.object.visible"].to_pylist(), bool)
        anchored, lines = anchor_object(pose, visible, touch, hull, stable_ups(hull), plane[:3], plane[3], support_tops(f"{src}/{d['support_surface']}"))
        human, trans = filter_human(np.asarray(t["observation.human.pose"].to_pylist(), np.float64), np.asarray(t["observation.human.translation"].to_pylist(), np.float64))
        t = column(t, "observation.object.pose", anchored)
        t = column(t, "observation.human.pose", human)
        t = column(t, "observation.human.translation", trans)
        t = t.set_column(t.schema.get_field_index("observation.object.visible"), t.schema.field("observation.object.visible"), pa.chunked_array([pa.array(np.ones(len(pose), bool))]))
        pq.write_table(t.replace_schema_metadata(t.schema.metadata), f"{dst}/data/chunk-000/{os.path.basename(path)}")
        shift = np.linalg.norm(anchored[visible, :3] - pose[visible, :3], axis=1)
        angle = np.degrees((rot_of(anchored[visible, 3:]) * rot_of(pose[visible, 3:]).inv()).magnitude())
        print(f"ep{ep} {d['object'][:18]}: moved mean {100 * shift.mean():.1f} max {100 * shift.max():.1f} cm, turned mean {angle.mean():.0f} max {angle.max():.0f} deg | " + "; ".join(lines), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4:]])
