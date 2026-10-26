"""Compare training runs from their metrics.jsonl (written by train/train.py) over a step window, before they finish.

    python evaluate/compare_runs.py <start_step> <end_step> <label>=<run_dir> ...

Prints the window mean of the key metrics for each run. After 210M steps the VOC assistance is 0 and these are
unassisted rollouts (with the training reset distribution); before that, compare runs only at equal steps.
"""

import json
import sys
from pathlib import Path

import numpy as np

METRICS = {
    "object pos err (m)": "tracking_error/object_position_m_mean_env_max_per_step",
    "object ori err (rad)": "tracking_error/object_orientation_rad_mean_env_max_per_step",
    "reaches end": "termination/reference_end_fraction_per_completed_episode",
    "term: object": "termination/object_fraction_per_completed_episode",
    "term: palm": "termination/palm_fraction_per_completed_episode",
    "term: pelvis": "termination/pelvis_fraction_per_completed_episode",
    "progress": "episode/reference_progress_mean_per_completed_episode",
    "voc": "curriculum/voc_scale",
}

start, end = int(sys.argv[1]), int(sys.argv[2])
runs = [arg.split("=", 1) for arg in sys.argv[3:]]
print(f"{'run':24s}{'rows':>6s}" + "".join(f"{name:>22s}" for name in METRICS))
for label, run_dir in runs:
    path = Path(run_dir) / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []
    rows = [row for row in rows if start <= row["step"] <= end]
    values = [np.mean([row[key] for row in rows if key in row]) if rows else float("nan") for key in METRICS.values()]
    print(f"{label:24s}{len(rows):>6d}" + "".join(f"{value:>22.4f}" for value in values))
