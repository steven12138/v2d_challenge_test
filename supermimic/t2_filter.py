"""Denoise the Tier 2 input before retargeting, using Tier 2 alone (Tier 1 is never read).

The corruption is mostly a slow random walk (t2_noise.py: 75-98 % of the wobble of a resting object is below 1 Hz), so a low-pass filter cannot remove it.
What can: a physical prior. An object that nobody moves is still, so segments where it stays within a small ball are held at their median pose. The rest gets a
zero-phase low-pass that only removes jitter above what the motion contains. The human gets the same low-pass on translation and joint rotations.
usage: python -m supermimic.t2_filter <tier_2 dir> <output dir> [episode ...]    other dataset folders are symlinked
"""

import glob
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R

FPS = 30.0
OBJECT_CUTOFF_HZ = 5.0
HUMAN_CUTOFF_HZ = 6.0
STILL_RADIUS_M = 0.02  # displacement over +-0.5 s below this: not moved (the walk alone gives about 0.7 cm over 1 s)
STILL_MIN_FRAMES = 45
BLEND_FRAMES = 8


def lowpass(x, cutoff):
    b, a = butter(3, cutoff / (FPS / 2), "low")
    return filtfilt(b, a, x, axis=0, padlen=min(3 * max(len(a), len(b)), len(x) - 1))


def continuous_quat(q):
    """Sign-aligned quaternions, so that they can be filtered and averaged."""
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def fill_dropouts(pos, quat, visible):
    """Object pose on invisible frames: linear and normalized-linear between visible neighbours, held at the ends."""
    idx, good = np.arange(len(pos)), np.flatnonzero(visible)
    p = np.stack([np.interp(idx, good, pos[good, i]) for i in range(3)], -1)
    q = continuous_quat(quat[good])
    q = np.stack([np.interp(idx, good, q[:, i]) for i in range(4)], -1)
    return p, q / np.linalg.norm(q, axis=-1, keepdims=True)


def still_segments(pos):
    half = int(FPS / 2)
    n = len(pos)
    moved = np.linalg.norm(pos[np.minimum(np.arange(n) + half, n - 1)] - pos[np.maximum(np.arange(n) - half, 0)], axis=1)
    still = moved < STILL_RADIUS_M
    segments, start = [], None
    for i, s in enumerate(np.r_[still, False]):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if i - start >= STILL_MIN_FRAMES:
                segments.append((start, i))
            start = None
    return segments


def filter_object(pose, visible):
    """pose: [x, y, z, qw, qx, qy, qz] per frame. Returns the filtered pose and the still segments."""
    pos, quat = fill_dropouts(pose[:, :3], pose[:, 3:], visible)
    segments = still_segments(pos)
    held_p, held_q, weight = pos.copy(), quat.copy(), np.zeros(len(pos))
    for a, b in segments:
        held_p[a:b] = np.median(pos[a:b], axis=0)
        mean = quat[a:b].mean(0)
        held_q[a:b] = mean / np.linalg.norm(mean)
        ramp = np.minimum(np.minimum(np.arange(b - a) + 1, np.arange(b - a)[::-1] + 1) / BLEND_FRAMES, 1.0)
        weight[a:b] = ramp
    smooth_p = lowpass(pos, OBJECT_CUTOFF_HZ)
    smooth_q = lowpass(quat, OBJECT_CUTOFF_HZ)
    w = weight[:, None]
    p = w * held_p + (1 - w) * smooth_p
    q = w * held_q + (1 - w) * smooth_q
    q = q * np.sign((q * quat).sum(-1, keepdims=True) + 1e-9)
    return np.concatenate([p, q / np.linalg.norm(q, axis=-1, keepdims=True)], -1), segments


def filter_human(pose, translation):
    """pose: local rotation vectors (T, 3 * joints)."""
    T, J = pose.shape[0], pose.shape[1] // 3
    q = R.from_rotvec(pose.reshape(-1, 3)).as_quat().reshape(T, J, 4)
    out = np.zeros_like(pose)
    for j in range(J):
        s = lowpass(continuous_quat(q[:, j]), HUMAN_CUTOFF_HZ)
        out[:, 3 * j : 3 * j + 3] = R.from_quat(s / np.linalg.norm(s, axis=-1, keepdims=True)).as_rotvec()
    return out, lowpass(translation, HUMAN_CUTOFF_HZ)


def column(table, name, values):
    size = values.shape[1]
    arr = pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=pa.float32()), size)
    return table.set_column(table.schema.get_field_index(name), table.schema.field(name), pa.chunked_array([arr]))


def main(src, dst, episodes):
    os.makedirs(f"{dst}/data/chunk-000", exist_ok=True)
    for name in os.listdir(src):
        if name != "data" and not os.path.exists(f"{dst}/{name}"):
            os.symlink(os.path.abspath(f"{src}/{name}"), f"{dst}/{name}")
    for path in sorted(glob.glob(f"{src}/data/chunk-000/episode_*.parquet")):
        ep = int(os.path.basename(path)[8:14])
        if episodes and ep not in episodes:
            continue
        t = pq.read_table(path)
        pose = np.asarray(t["observation.object.pose"].to_pylist(), np.float64)
        visible = np.asarray(t["observation.object.visible"].to_pylist(), bool)
        filtered, segments = filter_object(pose, visible)
        human, trans = filter_human(np.asarray(t["observation.human.pose"].to_pylist(), np.float64), np.asarray(t["observation.human.translation"].to_pylist(), np.float64))
        t = column(t, "observation.object.pose", filtered)
        t = column(t, "observation.human.pose", human)
        t = column(t, "observation.human.translation", trans)
        t = t.set_column(t.schema.get_field_index("observation.object.visible"), t.schema.field("observation.object.visible"), pa.chunked_array([pa.array(np.ones(len(pose), bool))]))
        pq.write_table(t.replace_schema_metadata(t.schema.metadata), f"{dst}/data/chunk-000/{os.path.basename(path)}")
        changed = np.linalg.norm(filtered[visible, :3] - pose[visible, :3], axis=1)
        print(f"ep{ep}: {len(pose)} frames, still {sum(b - a for a, b in segments)} frames in {len(segments)} segments, mean shift {100 * changed.mean():.2f} cm max {100 * changed.max():.2f} cm", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]])
