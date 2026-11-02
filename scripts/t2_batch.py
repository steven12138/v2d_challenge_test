"""Queue lines of Tier 2 runs: t2_batch.py <gpu> <run tag> <motion root name> <episode> [...]"""
import json
import sys

gpu, tag, motion = sys.argv[1:4]
STAGES = " ".join(f"curriculum.stages.{i}.objective_weights.palm_orientation=0" for i in range(9))  # the human's noise makes that target unreliable
meta = {}
for line in open("/home/steven/project/track2/data/raw/tier_2_synthetic_noise/meta/episodes_metadata.jsonl"):
    d = json.loads(line)
    meta[d["episode_index"]] = d["sequence_id"]
for e in sys.argv[4:]:
    print(f"MOTION=$HOME/track2/data/retargeted/{motion} TRACK=track_2_tier2 FRAME_REF=$HOME/track2/data/raw/tier_2_anchored "
          f"bash $HOME/track2/supermimic/run_variant.sh {tag} {e} {meta[int(e)]} {gpu} {STAGES}")
