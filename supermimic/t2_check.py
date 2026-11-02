"""Does a denoised Tier 2 object pose get closer to the truth? Checked without any clean data (Tier 1 is never read).

A rigid grasp keeps the object fixed in the palm, and the human carries its own independent noise, so removing object noise must shrink the spread of the object in
the palm frame during holds. The palm comes from the retargeted motion, which is the same for every input compared here.
usage: python -m supermimic.t2_check <palm motion root> <tier_2 dir> [<denoised dir> ...]    prints spread in deg and cm per dataset, per episode and on average
"""

import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

from supermimic.t2_filter import fill_dropouts
from supermimic.t2_grasp_prior import clean_labels, runs

MIN_HOLD = 45


def source_pose(root, ep):
    t = pq.read_table(f"{root}/data/chunk-000/episode_{ep:06d}.parquet", columns=["observation.object.pose", "observation.object.visible"])
    pose = np.asarray(t["observation.object.pose"].to_pylist(), float)
    return fill_dropouts(pose[:, :3], pose[:, 3:], np.asarray(t["observation.object.visible"].to_pylist(), bool))


def spread(obj_p, obj_r, palm_p, palm_r, holds):
    """Time-weighted RMS of the object around its mean pose in the palm frame, in deg and cm."""
    deg, cm, frames = 0.0, 0.0, 0
    for h, a, b in holds:
        rel_r = palm_r[h][a:b].inv() * obj_r[a:b]
        rel_p = palm_r[h][a:b].inv().apply(obj_p[a:b] - palm_p[h][a:b])
        deg += (np.degrees((rel_r.mean().inv() * rel_r).magnitude()) ** 2).sum()
        cm += (10000 * ((rel_p - rel_p.mean(0)) ** 2).sum(-1)).sum()
        frames += b - a
    return np.sqrt(deg / max(frames, 1)), np.sqrt(cm / max(frames, 1)), frames


def main(palm_root, roots):
    meta = {}
    with open(f"{roots[0]}/meta/episodes_metadata.jsonl") as f:
        for line in f:
            d = json.loads(line)
            meta[d["episode_index"]] = d
    print("ep object hold frames | deg, cm of each dataset: " + " ".join(os.path.basename(r.rstrip("/")) for r in roots))
    total = np.zeros((len(roots), 2))
    count = 0
    for ep, d in sorted(meta.items()):
        row = pq.read_table(f"{palm_root}/whole_body/soma/sequence_id={d['sequence_id']}/robot_name=g1/data.parquet", columns=["ee_pose_w", "hand_contact_active", "object_body_position", "object_body_wxyz"]).to_pydict()
        ee = np.asarray(row["ee_pose_w"][0], float)
        labels = np.asarray(row["hand_contact_active"][0], float) > 0.5
        palm_p = [ee[:, h, :3] for h in range(2)]
        palm_r = [R.from_quat(ee[:, h, [4, 5, 6, 3]]) for h in range(2)]
        holds = [(h, a, b) for h in range(2) for a, b in runs(clean_labels(labels[h])) if b - a >= MIN_HOLD]
        if not holds:
            continue
        # one rigid transform from the source world into the G1 world, from frame 0 of the motion the palm belongs to
        p0, q0 = source_pose(roots[-1], ep)
        g_r = R.from_quat(np.asarray(row["object_body_wxyz"][0], float)[0, 0][[1, 2, 3, 0]]) * R.from_quat(q0[0][[1, 2, 3, 0]]).inv()
        g_p = np.asarray(row["object_body_position"][0], float)[0, 0] - g_r.apply(p0[0])
        parts = []
        for i, root in enumerate(roots):
            p, q = source_pose(root, ep)
            deg, cm, frames = spread(g_r.apply(p) + g_p, g_r * R.from_quat(q[:, [1, 2, 3, 0]]), palm_p, palm_r, holds)
            total[i] += deg, cm
            parts.append(f"{deg:5.1f} {cm:5.1f}")
        count += 1
        print(f"{ep:2d} {d['object'][:18]:18s} {frames:5d} | " + " | ".join(parts), flush=True)
    print("mean" + " " * 25 + "| " + " | ".join(f"{a / count:5.1f} {b / count:5.1f}" for a, b in total))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
