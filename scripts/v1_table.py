"""Official results of this version only: the averaged policy of each clip, over its repeats, and the mean over the clips done so far.
usage: python v1_table.py <logs_dir>
"""

import glob
import json
import os
import re
import sys

logs = sys.argv[1]
rows = []
for folder in sorted(glob.glob(f"{logs}/kp_ep*")):
    m = re.fullmatch(r"kp_ep(\d+)", os.path.basename(folder))
    if not m:
        continue
    runs = []
    for path in glob.glob(f"{folder}/eval_repeat_swa_last3/results_*/episode_*.json"):
        with open(path) as fh:
            x = json.load(fh)["metrics"]
        runs.append((x["add_auc"], x["spider_sr_uncentered"], x["maniptrans_sr"], x["mppe_cm"]))
    if runs:
        rows.append((int(m.group(1)), len(runs), [sum(r[i] for r in runs) / len(runs) for i in range(4)]))
print("episode runs AUC SP MP MPPE_cm")
for ep, n, v in sorted(rows):
    print(f"ep{ep:02d} {n} {v[0]:.3f} {v[1]:.2f} {v[2]:.2f} {v[3]:.2f}")
if rows:
    print(f"mean over {len(rows)}/30 clips: AUC {sum(r[2][0] for r in rows) / len(rows):.3f} SP {sum(r[2][1] for r in rows) / len(rows):.3f} "
          f"MP {sum(r[2][2] for r in rows) / len(rows):.3f} MPPE {sum(r[2][3] for r in rows) / len(rows):.2f}")
