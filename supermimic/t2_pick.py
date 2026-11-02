"""Choose among refined Tier 2 policies by Tier 2 metrics only: 64-world AUC against the policy input, minus the fall rate, since one fall costs a whole clip.
The official local score reads the clean reference, so it never enters this choice (PLAN.md 13.22.8).
usage: python -m supermimic.t2_pick <refinement run dir> [...]    prints the best policy path, and its numbers on stderr
"""

import glob
import os
import re
import sys


def main(dirs):
    best = None
    for run in dirs:
        for stats in glob.glob(f"{run}/hands/policy_ema_*.stats"):
            with open(stats) as f:
                text = f.read()
            auc, fell = re.search(r" AUC ([\d.]+)", text), re.search(r"fell ([\d.]+)", text)
            if not auc or not fell:
                continue
            score = float(auc.group(1)) - float(fell.group(1))
            path = f"{run}/checkpoints/{os.path.basename(stats)[:-6]}.safetensors"
            if os.path.exists(path) and (best is None or score > best[0]):
                best = (score, path, auc.group(1), fell.group(1))
    if best is None:
        raise SystemExit("no statistics found")
    print(f"score {best[0]:.3f}: AUC {best[2]} fell {best[3]}", file=sys.stderr)
    print(best[1])


if __name__ == "__main__":
    main(sys.argv[1:])
