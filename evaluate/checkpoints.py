"""Write the checkpoint map that the official kit's examples/run_policy_evaluation.py takes.

    python evaluate/checkpoints.py <dataset_dir> <train_dir> <out.json>

The kit identifies episodes by index, so this maps each episode of <dataset_dir> to the final policy
in <train_dir>/<sequence_id>/. Episodes without a final policy are reported and left out.
"""

import json
import sys
from pathlib import Path

FINAL_POLICY = "policy_349999104.safetensors"

dataset_dir, train_dir, out_json = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

checkpoints, missing = {}, []
for line in open(dataset_dir / "meta" / "episodes_metadata.jsonl"):
    episode = json.loads(line)
    policy = train_dir / episode["sequence_id"] / FINAL_POLICY
    if policy.is_file():
        checkpoints[str(episode["episode_index"])] = str(policy.resolve())
    else:
        missing.append(episode["sequence_id"])

out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(checkpoints, indent=2) + "\n")
print(f"{len(checkpoints)} checkpoints -> {out_json}")
if missing:
    print(f"no final policy yet for {len(missing)} sequences: {missing}")
