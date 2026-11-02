"""One-line per-phase summary of an evaluator object trajectory (ep 8 phases: approach / contact / push / hold)."""

import sys

import numpy as np
import pyarrow.parquet as pq

PHASES = ((0, 300, "approach"), (300, 411, "contact"), (411, 600, "push"), (600, 708, "hold"))


def main(path: str, episode: int = 8) -> None:
    t = pq.read_table(path).to_pydict()
    a = np.stack([t["achieved_position_" + c] for c in "xyz"], -1)
    r = np.stack([t["reference_position_" + c] for c in "xyz"], -1)
    aq = np.stack([t["achieved_quaternion_" + c] for c in "xyzw"], -1)
    rq = np.stack([t["reference_quaternion_" + c] for c in "xyzw"], -1)
    e = np.linalg.norm(a - r, axis=1)
    ang = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(aq * rq, 1)), 0, 1)))
    parts = []
    n = len(e)
    phases = PHASES if episode == 8 else tuple((i * n // 4, (i + 1) * n // 4, f"q{i + 1}") for i in range(4))
    for lo, hi, name in phases:
        s = slice(lo, min(hi, len(e)))
        parts.append(f"{name} {100 * e[s].mean():.1f}cm/{ang[s].mean():.0f}deg")
    tipped = np.nonzero(ang > 60)[0]
    tip = f"tipped@{int(tipped[0])}" if len(tipped) else "upright"
    print(f"SEGMENTS mean {100 * e.mean():.1f}cm/{ang.mean():.0f}deg | " + " | ".join(parts) + f" | {tip}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8)
