"""Diagnostic: tracking errors of pure SONIC (zero residual) 51 steps after an explicit reset at several frames."""

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
        tm = env.termination
        F = handles.reference.num_frames
        zero = wp.zeros(env.action_input.shape, dtype=wp.float32)
        for start in range(0, F - 60, max(1, (F - 60) // 16)):
            env.reset(frame_id=start)
            worst = np.zeros(4)
            for step in range(51):
                env.step(zero)
                wp.synchronize_device()
                errs = np.array([float(tm.pelvis_position_error.numpy()[0]), float(tm.palm_position_error.numpy()[0]),
                                 float(tm.object_position_error.numpy()[0]),
                                 float(tm.object_orientation_error.numpy()[0])])
                worst = np.maximum(worst, errs)
            print(f"start {start:4d}: after 51 steps pelvis {100 * errs[0]:5.1f} cm  palm {100 * errs[1]:5.1f} cm  "
                  f"object {100 * errs[2]:5.1f} cm / {np.degrees(errs[3]):5.1f} deg | max palm {100 * worst[1]:5.1f} cm")


if __name__ == "__main__":
    main(sys.argv[1])
