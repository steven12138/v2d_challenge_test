"""The SOMA-X human (and its object poses) in the G1 world of the official retargeter.

Applies the transforms of retarget/soma_to_g1.py, using the same robotic_grounding functions: frame-0
anchoring of the body, ground-plane levelling, then SOMA -> robot axes. The only step left out is
soma_to_g1's small constant object lift (at most 1 cm).

    from human_world import human_in_robot_world
    human = human_in_robot_world(Path("<soma_dir>/sequence_id=<seq>"))
    human["joints"]  (T, 77, 3)   human["joint_names"]  (77,)   human["vertices"]  (T, V, 3)
    human["faces"]   (F, 3)       human["object_poses"] (T, 4, 4) world_T_object
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "robotic_grounding"))

import numpy as np
import torch

from robotic_grounding.retarget.ground_alignment import compute_plane_leveling_transform, load_ground_plane_robot_frame
from robotic_grounding.retarget.read_soma import SOMA
from robotic_grounding.retarget.robot_config import load_robot_config

R_CV_TO_SOMA = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV world -> SOMA world, as in soma_to_g1.py


def human_in_robot_world(folder: Path) -> dict:
    # Human and object poses in the frame-0 anchored SOMA world.
    soma = SOMA(identity_model_type="mhr", device=torch.device("cpu"))
    motion = soma.load_motion(folder / "soma_params.npz", normalize=True)
    anchor = np.eye(4)
    anchor[:3, :3] = motion["first_frame_R_inv"]
    anchor[:3, 3] = -motion["first_frame_R_inv"] @ motion["first_frame_transl"]
    object_poses = anchor @ R_CV_TO_SOMA @ np.load(folder / "poses.npy")

    # Level the fitted ground plane to z=0, expressed in SOMA axes.
    r_world = np.asarray(load_robot_config("g1").r_world, dtype=np.float64)
    plane = load_ground_plane_robot_frame(
        folder / "ground_plane.json", cv_to_source=R_CV_TO_SOMA, first_frame_anchor=anchor, source_to_robot=r_world)
    level_rotation, level_translation = compute_plane_leveling_transform(plane)
    level = np.eye(4)
    level[:3, :3] = r_world.T @ level_rotation @ r_world
    level[:3, 3] = r_world.T @ level_translation

    # Everything into the robot world (Z up).
    to_robot = np.eye(4)
    to_robot[:3, :3] = r_world
    world = to_robot @ level
    return {
        "joints": motion["joints"] @ world[:3, :3].T + world[:3, 3],
        "joint_names": np.array(motion["joint_names"]),
        "vertices": motion["vertices"] @ world[:3, :3].T + world[:3, 3],
        "faces": np.asarray(soma.faces),
        "object_poses": world @ object_poses,
    }
