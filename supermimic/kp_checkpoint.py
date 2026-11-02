"""Rewrite a checkpoint trained with the object keypoint reward so the pristine FlashCHORD evaluator can rebuild its environment.

The evaluator rebuilds the environment from the config stored in the checkpoint, and the keypoint reward exists only on Grace's modified
flash_chord or in supermimic.flash_objective. Drops the term from the config, restores the official objective class and points the motion
at the H100 copy (the same Pink IK motion); the weights stay byte-identical, and rewards are not used in evaluation.
usage: python -m supermimic.kp_checkpoint <policy.safetensors> <out.safetensors>
"""

import json
import struct
import sys

GRACE_MOTION = "/scratch/user/the/v2d/t2/work/tier1/assets/human_motion_data/"
H100_MOTION = "/data/the/nvidia_challenge_track2/retargeted/t1/"
GRACE_GRASP_MOTION = "/scratch/user/the/v2d/t2/work/tier1/grasp/assets/human_motion_data/"  # hand-refined motion (dex3_grasp.py)
H100_GRASP_MOTION = "/home/the/track2/data/retargeted_grasp/t1/"
CONFIG_KEY = "user.resolved_config_json"
OFFICIAL_OBJECTIVE = "flash_chord.objectives.recon_body.ReconBodyObjectiveConfig"


def official_config(config: dict) -> dict:
    for section in (config, config["training"]):
        for stage in section["curriculum"]["stages"]:
            stage["objective_weights"].pop("object_keypoints", None)
    for objective in (config["objective"], config["env"]["objective"]):
        objective["_target_"] = OFFICIAL_OBJECTIVE
        for name in [n for n in objective if n.startswith(("object_keypoints", "finger_grip", "action_rate_extra", "action_l2_extra"))]:
            del objective[name]
    return config


def convert(src: str, dst: str) -> None:
    with open(src, "rb") as fh:
        size = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(size))
        weights = fh.read()
    meta = header["__metadata__"]
    config = official_config(json.loads(meta[CONFIG_KEY]))
    meta[CONFIG_KEY] = json.dumps(config, sort_keys=True).replace(GRACE_MOTION, H100_MOTION).replace(GRACE_GRASP_MOTION, H100_GRASP_MOTION)
    if any(left in meta[CONFIG_KEY] for left in ("keypoints", "finger_grip", "supermimic", GRACE_MOTION, GRACE_GRASP_MOTION)):
        raise ValueError(f"{src}: training-only settings are left in the config")
    encoded = json.dumps(header, separators=(",", ":")).encode()
    with open(dst, "wb") as fh:
        fh.write(struct.pack("<Q", len(encoded)))
        fh.write(encoded)
        fh.write(weights)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
