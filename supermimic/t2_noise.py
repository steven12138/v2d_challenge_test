"""What does the Tier 2 corruption look like, from Tier 2 alone (Tier 1 is never read)?

Per episode: dropped frames, and the object's motion on segments where it must be still: it starts at rest and the human is far from it. Their wobble
is pure input noise, so its size and spectrum say which filter can help: high-frequency jitter or a slow random walk.
usage: python -m supermimic.t2_noise <tier_2 dataset dir> [episode ...]
"""

import glob
import os
import sys

import numpy as np
import pyarrow.parquet as pq

FPS = 30.0


def angle_deg(a, b):
    d = np.abs(np.sum(a * b, -1)) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)
    return np.degrees(2 * np.arccos(np.clip(d, 0, 1)))


def analyze(path):
    t = pq.read_table(path)
    pose = np.asarray(t["observation.object.pose"].to_pylist(), np.float64)
    vis = np.asarray(t["observation.object.visible"].to_pylist(), bool)
    root = np.asarray(t["observation.human.translation"].to_pylist(), np.float64)
    p, q = pose[:, :3], pose[:, 3:]
    good = np.flatnonzero(vis)
    dp = np.linalg.norm(np.diff(p[good], axis=0), axis=1)
    dq = angle_deg(q[good][1:], q[good][:-1])
    # the rest segment: from the start while the object stays within a small ball, before the human is near it
    dist = np.linalg.norm(root[good] - p[good], axis=1)
    ball = np.linalg.norm(p[good] - np.median(p[good][:15], axis=0), axis=1)
    end = next((i for i in range(len(good)) if ball[i] > 0.08), len(good))
    n = max(end - 15, 0)
    rest = p[good][:n]
    out = dict(frames=len(p), dropped=int((~vis).sum()), step_cm=100 * np.median(dp), step_p99_cm=100 * np.percentile(dp, 99), step_deg=float(np.median(dq)), rest=n, dist0=float(dist[0]))
    if n >= 30:
        e = rest - rest.mean(0)
        out["rest_rms_cm"] = 100 * np.sqrt((e**2).sum(-1).mean())
        out["rest_step_cm"] = 100 * np.median(np.linalg.norm(np.diff(rest, axis=0), axis=1))
        spec = (np.abs(np.fft.rfft(e, axis=0)) ** 2).sum(-1)
        freq = np.fft.rfftfreq(n, 1 / FPS)
        out["rest_lt1hz"] = float(spec[(freq > 0) & (freq < 1)].sum() / max(spec[freq > 0].sum(), 1e-12))
        out["rest_gt5hz"] = float(spec[freq >= 5].sum() / max(spec[freq > 0].sum(), 1e-12))
        out["rest_deg"] = float(np.median(angle_deg(q[good][:n], q[good][:n][n // 2])))
        # random walk: rms displacement grows with lag; white noise stays flat
        out["growth"] = float(np.sqrt(((rest[20:] - rest[:-20]) ** 2).sum(-1).mean()) / max(np.sqrt(((rest[1:] - rest[:-1]) ** 2).sum(-1).mean()), 1e-9))
    return out


def main(root, eps):
    files = sorted(glob.glob(f"{root}/data/chunk-000/episode_*.parquet"))
    print("ep frames dropped | step cm med p99, deg med | rest frames, rms cm, step cm, deg, <1Hz, >5Hz, growth(lag20/lag1; white ~1, walk ~4.5)")
    for f in files:
        e = int(os.path.basename(f)[8:14])
        if eps and e not in eps:
            continue
        o = analyze(f)
        print(f"{e:2d} {o['frames']:5d} {o['dropped']:4d} | {o['step_cm']:.2f} {o['step_p99_cm']:.2f} {o['step_deg']:.2f} | {o['rest']:4d} "
              + (f"{o['rest_rms_cm']:.2f} {o['rest_step_cm']:.2f} {o['rest_deg']:.2f} {o['rest_lt1hz']:.2f} {o['rest_gt5hz']:.2f} {o['growth']:.1f}" if "rest_rms_cm" in o else "-"), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]])
