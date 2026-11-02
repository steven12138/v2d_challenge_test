"""Best checkpoint per episode from every official evaluation under a directory, and the map the kit's evaluator takes.

Groups the kit's episode_XXXXXX.json files by checkpoint hash, averages their metrics over repeats, and ranks checkpoints by
AUC + SP + MP - MPPE / 100, the four scored metrics on comparable scales. A checkpoint chosen here gets one fresh official run for the
submission, so that no lucky repeat is selected.
usage: python scan_official.py <logs_dir> <out_checkpoints.json>
"""

import json
import os
import re
import sys
from collections import defaultdict

EPISODES = range(30)


def main(logs: str, out: str) -> None:
    runs = defaultdict(list)
    for root, _, files in os.walk(logs, followlinks=True):
        if re.search(r"_t2[a-z]*(/|$)", root):  # Tier 2 runs never enter a Tier 1 selection
            continue
        for name in files:
            m = re.fullmatch(r"episode_(\d{6})\.json", name)
            if not m:
                continue
            try:
                with open(os.path.join(root, name)) as fh:
                    d = json.load(fh)
                x = d["metrics"]
                row = (x["add_auc"], x["spider_sr_uncentered"], x["maniptrans_sr"], x["mppe_cm"])
            except (OSError, ValueError, KeyError):
                continue
            runs[(int(m.group(1)), d.get("checkpoint_sha256", d["checkpoint"]))].append((d["checkpoint"], row))
    best = {}
    print("ep runs AUC SP MP MPPE checkpoint")
    for ep in EPISODES:
        rows = []
        for (e, _), entries in runs.items():
            path = next((p for p, _ in entries if os.path.exists(p)), None)
            if e != ep or path is None:
                continue
            n = len(entries)
            mean = [sum(r[i] for _, r in entries) / n for i in range(4)]
            rows.append((mean[0] + mean[1] + mean[2] - min(mean[3], 200.0) / 100.0, n, mean, path))
        if not rows:
            print(f"{ep:2d} none")
            continue
        _, n, mean, path = max(rows, key=lambda r: r[0])
        best[ep] = path
        print(f"{ep:2d} {n:2d} {mean[0]:.3f} {mean[1]:.2f} {mean[2]:.2f} {mean[3]:6.2f} {path.replace(logs, '')[-62:]}")
    if best:
        vals = [[sum(r[i] for _, r in runs[k]) / len(runs[k]) for i in range(4)] for k in runs if k[0] in best and best[k[0]] in [p for p, _ in runs[k]]]
        print(f"episodes covered {len(best)}/30; mean of the selected AUC {sum(v[0] for v in vals) / len(vals):.3f} SP {sum(v[1] for v in vals) / len(vals):.3f} "
              f"MP {sum(v[2] for v in vals) / len(vals):.3f} MPPE {sum(v[3] for v in vals) / len(vals):.2f}")
    with open(out, "w") as fh:
        json.dump({str(e): p for e, p in sorted(best.items())}, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
