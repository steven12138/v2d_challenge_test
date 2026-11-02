"""Diagnostic: gap between the reference palm pose and the palm FK of the reference joint pose.

Resets 1 world explicitly to a set of frames (robot exactly at the reference joint pose) and reads observation
block 11 (future palm position delta, offset 0) and block 12 (palm orientation delta), which then equal
`reference_palm - FK(reference_q)`. Palm termination fires at 0.15 m / 1.5 rad after the 50-step mask.
"""

import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import warp as wp

from supermimic import env_io


def main(parquet: str) -> None:
    with wp.ScopedDevice("cuda:0"):
        handles = env_io.build(parquet, 1)
        env = handles.env
        F = handles.reference.num_frames
        rows = []
        for frame in range(0, F, max(1, F // 40)):
            env.reset(frame_id=frame)
            obs = env.observation.numpy().reshape(-1)
            cursor = 6 + 12 + 6 + 3 + 6 + 43 + 43 + 9 + 18 + 129
            palm_delta = obs[cursor : cursor + 18].reshape(3, 2, 3)[0]  # offset 0, [right, left]
            rows.append((frame, np.linalg.norm(palm_delta, axis=-1)))
        worst = max(r[1].max() for r in rows)
        for frame, d in rows:
            print(f"frame {frame:4d}  palm gap right {100 * d[0]:6.1f} cm  left {100 * d[1]:6.1f} cm")
        print(f"MAX palm gap over frames: {100 * worst:.1f} cm (termination threshold 15 cm)")


if __name__ == "__main__":
    main(sys.argv[1])
