"""Write the checkpoint map that the official kit's examples/run_policy_evaluation.py takes.

    python evaluate/checkpoints.py <dataset_dir> <train_dir> <out.json> [newest]

The kit identifies episodes by index, so this maps each episode of <dataset_dir> to the final policy in
<train_dir>/<sequence_id>/. With "newest", a sequence without a final policy uses its newest policy checkpoint
instead (for runs stopped early). Episodes without a usable policy are reported and left out.
"""

import json
import sys
from pathlib import Path

FINAL_POLICY = "policy_349999104.safetensors"

dataset_dir, train_dir, out_json = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
newest = len(sys.argv) > 4 and sys.argv[4] == "newest"

checkpoints, missing, early = {}, [], []
for line in open(dataset_dir / "meta" / "episodes_metadata.jsonl"):
    episode = json.loads(line)
    run = train_dir / episode["sequence_id"]
    policy = run / FINAL_POLICY
    if not policy.is_file() and newest:
        candidates = sorted(run.glob("policy_*.safetensors"), key=lambda path: int(path.stem.split("_")[1]))
        if candidates:
            policy = candidates[-1]
            early.append(f"{episode['sequence_id'][20:]}@{int(policy.stem.split('_')[1]) // 1_000_000}M")
    if policy.is_file():
        checkpoints[str(episode["episode_index"])] = str(policy.resolve())
    else:
        missing.append(episode["sequence_id"])

out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(checkpoints, indent=2) + "\n")
print(f"{len(checkpoints)} checkpoints -> {out_json}")
if early:
    print(f"{len(early)} not final: {early}")
if missing:
    print(f"no policy for {len(missing)} sequences: {missing}")
