"""Kinematic quality of a retargeted G1 motion, before any RL: compare retargeters without training.

    python retarget/eval_retarget.py <soma_dir>/sequence_id=<seq> <motion data.parquet> <g1_dex3_w_object.xml> <out.json>

The scene comes from retarget/omniretarget/build_scene.py: the official G1 + Dex3 (all 43 joints, as trained) plus
the CoACD-decomposed object. All robot-object and robot-ground distances are MuJoCo signed distances
(negative = penetration). Metrics follow OmniRetarget's evaluation (holosoma evaluation/eval_retargeting.py):
  penetration     fraction of frames whose deepest robot-object (robot-ground) penetration exceeds 1 cm,
                  and the mean / max of that per-frame depth (m); also without the Dex3 finger links, whose
                  joints the RL policy adjusts with its finger residual
  contact         per hand / foot, over the frames where the human point is within 3 cm of the object surface:
                  recall = fraction where the robot hand (palm + fingers) / foot is within 3 cm; mean distance (m)
  foot sliding    mean horizontal speed of the ankle-roll links while they are < 1 cm above the ground (m/s)
Ground penetration and foot height are exact geom-plane distances (mj_geomDistance), not read from the contact
list, and the run fails if MuJoCo's contact buffer overflows.
  joint limits    frames with any joint outside its range
  object          max distance between the motion's object path and the human_world one (sanity check, m)
  keyframes       up to 4 frames with the deepest object penetration (>= 0.5 cm), 45+ frames apart
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco
import numpy as np
import pyarrow.parquet as pq
import trimesh

from human_world import human_in_robot_world

HUMAN_POINTS = {"left_hand": "LeftHandMiddle2", "right_hand": "RightHandMiddle2",
                "left_foot": "LeftToeBase", "right_foot": "RightToeBase"}
FEET = {"left_foot": "left_ankle_roll_link", "right_foot": "right_ankle_roll_link"}
MARGIN = 0.05  # MuJoCo reports every geom pair closer than this
FPS = 30.0

folder, motion_path, scene_xml, out_json = (Path(arg) for arg in sys.argv[1:5])

row = pq.read_table(motion_path).to_pylist()[0]
joints = dict(zip(row["robot_joint_names"], np.asarray(row["robot_joint_positions"]).T))
object_pos = np.asarray(row["object_body_position"])[:, 0]
object_wxyz = np.asarray(row["object_body_wxyz"])[:, 0]
frames = len(object_pos)

model = mujoco.MjModel.from_xml_path(str(scene_xml))
data = mujoco.MjData(model)
hinge = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
qpos = np.concatenate([np.asarray(row["robot_root_position"]), np.asarray(row["robot_root_wxyz"]),
                       np.stack([joints[model.joint(j).name] for j in hinge], 1), object_pos, object_wxyz], axis=1)
lower, upper = model.jnt_range[hinge].T

body_names = [model.body(b).name for b in range(model.nbody)]
object_body = model.body("object_link").id
limb_of_body = {}
for b, name in enumerate(body_names):
    for side in ("left", "right"):
        if name.startswith(f"{side}_hand"):
            limb_of_body[b] = f"{side}_hand"
        if name == FEET[f"{side}_foot"]:
            limb_of_body[b] = f"{side}_foot"
foot_ids = [model.body(FEET["left_foot"]).id, model.body(FEET["right_foot"]).id]
ground_geom = model.geom("ground").id
robot_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] not in (0, object_body)]
foot_geoms = [[g for g in robot_geoms if model.geom_bodyid[g] == foot] for foot in foot_ids]
fromto = np.zeros(6)


def ground_distance(geoms):
    """Smallest signed distance between these geoms and the ground plane (negative = below ground)."""
    return min(mujoco.mj_geomDistance(model, data, g, ground_geom, 1.0, fromto) for g in geoms)

model.geom_margin[:] = MARGIN
object_pen, ground_pen, body_pen = np.zeros(frames), np.zeros(frames), np.zeros(frames)
is_finger = [("_hand_" in name and not name.endswith("palm_link")) for name in body_names]
limb_distance = {limb: np.full(frames, MARGIN) for limb in HUMAN_POINTS}
foot_grounded = np.zeros((frames, 2), dtype=bool)
foot_xy = np.zeros((frames, 2, 2))
limit_frames = 0
for frame in range(frames):
    data.qpos[:] = qpos[frame]
    mujoco.mj_forward(model, data)
    if data.warning[mujoco.mjtWarning.mjWARN_CONTACTFULL].number:
        raise RuntimeError(f"frame {frame}: MuJoCo contact buffer full, distances would be incomplete")
    for contact in data.contact[:data.ncon]:
        b1, b2 = model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]
        if (b1 == object_body) != (b2 == object_body) and 0 not in (b1, b2):  # robot-object (not ground-object)
            robot_body = b2 if b1 == object_body else b1
            object_pen[frame] = max(object_pen[frame], -contact.dist)
            if not is_finger[robot_body]:
                body_pen[frame] = max(body_pen[frame], -contact.dist)
            limb = limb_of_body.get(robot_body)
            if limb is not None:
                limb_distance[limb][frame] = min(limb_distance[limb][frame], contact.dist)
    ground_pen[frame] = max(0.0, -ground_distance(robot_geoms))
    foot_grounded[frame] = [ground_distance(geoms) < 0.01 for geoms in foot_geoms]
    foot_xy[frame] = data.xpos[foot_ids][:, :2]
    q = qpos[frame, 7:7 + len(hinge)]
    limit_frames += int(np.any((q < lower - 1e-3) | (q > upper + 1e-3)))

# Human points against the object surface (object frame).
human = human_in_robot_world(folder)
names = list(human["joint_names"])
mesh = trimesh.load(folder / "reconstructed_mesh" / "output_aligned.glb", force="mesh")
world_T_object = np.tile(np.eye(4), (frames, 1, 1))
for frame in range(frames):
    world_T_object[frame] = trimesh.transformations.quaternion_matrix(object_wxyz[frame])
    world_T_object[frame, :3, 3] = object_pos[frame]

contact = {}
for limb, human_joint in HUMAN_POINTS.items():
    points = human["joints"][:, names.index(human_joint)]
    local = np.einsum("tji,tj->ti", world_T_object[:, :3, :3], points - world_T_object[:, :3, 3])
    _, human_distance, _ = trimesh.proximity.closest_point(mesh, local)
    in_contact = human_distance < 0.03
    robot_distance = np.maximum(limb_distance[limb], 0.0)
    contact[limb] = {
        "human_contact_frames": int(in_contact.sum()),
        "recall": float((robot_distance[in_contact] < 0.03).mean()) if in_contact.any() else None,
        "mean_robot_distance_m": float(robot_distance[in_contact].mean()) if in_contact.any() else None,
    }

grounded = foot_grounded[1:] & foot_grounded[:-1]
speed = np.linalg.norm(np.diff(foot_xy, axis=0), axis=-1) * FPS
worst_frames = []
for frame in np.argsort(-object_pen):
    if object_pen[frame] < 0.005 or len(worst_frames) == 4:
        break
    if all(abs(int(frame) - other) >= 45 for other in worst_frames):
        worst_frames.append(int(frame))

result = {
    "motion": str(motion_path),
    "frames": frames,
    "object_penetration": {"fraction_over_1cm": float((object_pen > 0.01).mean()),
                           "mean_m": float(object_pen.mean()), "max_m": float(object_pen.max())},
    "object_penetration_without_fingers": {"fraction_over_1cm": float((body_pen > 0.01).mean()),
                                           "mean_m": float(body_pen.mean()), "max_m": float(body_pen.max())},
    "ground_penetration": {"fraction_over_1cm": float((ground_pen > 0.01).mean()),
                           "mean_m": float(ground_pen.mean()), "max_m": float(ground_pen.max())},
    "contact": contact,
    "foot_sliding_mps": float(speed[grounded].mean()) if grounded.any() else None,
    "joint_limit_frames": limit_frames,
    "object_path_vs_human_world_max_m": float(np.linalg.norm(human["object_poses"][:, :3, 3] - object_pos, axis=1).max()),
    "worst_object_penetration_frames": worst_frames,
}
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
