"""Retarget a SOMA-X sequence to G1 with OmniRetarget's interaction-mesh solver (arXiv 2509.26633, holosoma).
Replaces the official Pink-IK soma_to_g1.py: no part of its solution is used.

    python retarget/omniretarget/omniretarget.py <soma_dir>/sequence_id=<seq> <scene_dir> <motion_root>

<scene_dir> comes from build_scene.py. This writes the FlashCHORD training motion
    <motion_root>/whole_body/soma/sequence_id=<seq>/robot_name=g1/data.parquet
    <motion_root>/whole_body/soma/sequence_id=<seq>/object/   (textured_mesh.obj + .urdf, as soma_to_g1.py does)
(prep/prep.py already put the support stage in <motion_root>/whole_body/reconstructed_stage/).

  * Human and object go into the robot world with the transforms of soma_to_g1.py (retarget/human_world.py), plus
    its constant object ground lift (<= 1 cm). So the object trajectory is the official one up to a rigid offset,
    which the scorer's frame conversion requires.
  * Each frame solves OmniRetarget's SQP. It preserves the Laplacian coordinates of an interaction mesh of 15 body
    points, 100 object surface points and a 5 x 5 ground grid under the human root, all in the object frame,
    under hard constraints: object and ground non-penetration, foot sticking, and joint limits.
  * As in OmniRetarget, the demonstration side of the mesh is scaled by robot height / human height, and the robot
    side uses the real object. The ground points keep the feet on the ground: without them, scaling about the
    object rather than the ground leaves the robot floating. OmniRetarget instead scales the whole scene, object
    trajectory included, which the leaderboard does not allow.
  * The per-frame SQP solution jitters (leg joint acceleration ~6x the official retarget's), which the SONIC
    tracker cannot follow. So root position, root orientation and the 29 joints are low-passed with a zero-phase
    4th-order Butterworth at 3 Hz (filtfilt); this changes joints by ~0.01 rad RMS.
  * Dex3 finger joints stay at 0 (open), like the official retarget; the RL policy's finger residual moves them.
"""

import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "robotic_grounding"))

import numpy as np
import trimesh
import yourdfpy
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation

from human_world import human_in_robot_world
from interaction_mesh_retargeter import InteractionMeshRetargeter
from robotic_grounding.motion_schema import MotionData, save_motion_parquet
from robotic_grounding.retarget.ground_alignment import ReferencePlane, compute_object_ground_lift
from utils import extract_foot_sticking_sequence_velocity

G1_DEX3_URDF = HERE.parent / "robotic_grounding" / "robotic_grounding" / "assets" / "urdfs" / "g1" / "main_with_hand.urdf"
ROBOT_HEIGHT = 1.32  # holosoma's G1 height
PELVIS_HEIGHT = 0.78  # standing G1 pelvis height, for the first-frame initial guess only
FPS = 30.0
CONTACT_THRESHOLD = 0.02  # soma_to_g1.py --contact-threshold default

# Joint order of the official G1 + Dex3 motions (the URDF's own order differs).
G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_yaw_joint", "waist_roll_joint",
    "waist_pitch_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint", "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint", "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint", "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint", "right_hand_thumb_0_joint", "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]
EE_LINKS = ["left_hand_palm_link", "right_hand_palm_link"]

# SOMA joint -> G1 (sphere hand) body: holosoma's ("mocap", "g1") mapping with the SOMA joint names.
JOINTS_MAPPING = {
    "Spine1": "pelvis_contour_link",
    "LeftLeg": "left_hip_pitch_link",
    "LeftShin": "left_knee_link",
    "LeftToeBase": "left_ankle_roll_sphere_5_link",
    "RightLeg": "right_hip_pitch_link",
    "RightShin": "right_knee_link",
    "RightToeBase": "right_ankle_roll_sphere_5_link",
    "LeftArm": "left_shoulder_roll_link",
    "LeftForeArm": "left_elbow_link",
    "LeftHandMiddle2": "left_sphere_hand_link",
    "RightArm": "right_shoulder_roll_link",
    "RightForeArm": "right_elbow_link",
    "RightHandMiddle2": "right_sphere_hand_link",
    "LeftFoot": "left_ankle_intermediate_1_link",
    "RightFoot": "right_ankle_intermediate_1_link",
}
TOE_NAMES = ["LeftToeBase", "RightToeBase"]
SMOOTHING_CUTOFF_HZ = 3.0
GROUND_GRID = np.array([[x, y, 0.0] for x in np.linspace(-0.6, 0.6, 5) for y in np.linspace(-0.6, 0.6, 5)])

folder, scene_dir, motion_root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sequence_id = folder.name.removeprefix("sequence_id=")
sequence_dir = motion_root / "whole_body" / "soma" / f"sequence_id={sequence_id}"

# Object mesh, stored next to the motion as soma_to_g1.py does.
object_dir = sequence_dir / "object"
object_dir.mkdir(parents=True, exist_ok=True)
mesh = trimesh.load(folder / "reconstructed_mesh" / "output_aligned.glb", force="mesh")
mesh.export(object_dir / "textured_mesh.obj")
(object_dir / "textured_mesh.urdf").write_text("""<?xml version="1.0"?>
<robot name="retarget_object">
  <link name="object">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual><geometry><mesh filename="textured_mesh.obj"/></geometry></visual>
    <collision><geometry><mesh filename="textured_mesh.obj"/></geometry></collision>
  </link>
</robot>
""")
object_vertices = np.asarray(mesh.vertices)

# Human and object in the robot world, with soma_to_g1.py's constant object ground lift.
human = human_in_robot_world(folder)
human_joints = human["joints"]
joint_names = list(human["joint_names"])
frames = len(human_joints)
object_T = human["object_poses"].copy()
frame0_vertices = object_vertices @ object_T[0, :3, :3].T + object_T[0, :3, 3]
lift = compute_object_ground_lift(frame0_vertices, ReferencePlane.horizontal(), penetration_tolerance=5e-4,
                                  clearance=5e-4, max_lift=1e-2).applied_lift
object_T[:, 2, 3] += lift
object_pos = object_T[:, :3, 3]
object_wxyz = Rotation.from_matrix(object_T[:, :3, :3]).as_quat(scalar_first=True)
object_poses = np.concatenate([object_pos, object_wxyz], axis=1)  # [x, y, z, qw, qx, qy, qz], MuJoCo order

# Demonstration side scaled to the robot, in the object frame: scale positions of human and object alike.
scale = ROBOT_HEIGHT / float(np.ptp(human["vertices"][0][:, 2]))
object_points = np.asarray(trimesh.sample.sample_surface(mesh, 100, seed=42)[0])
demo_object_poses = object_poses.copy()
demo_object_poses[:, :3] *= scale

# Object surface points plus ground points under the human root, in the object frame of every frame.
root_xy = human_joints[:, joint_names.index("Hips"), :2]
points_local = []
for frame in range(frames):
    ground = GROUND_GRID + np.array([root_xy[frame, 0], root_xy[frame, 1], 0.0])
    points_local.append(np.vstack([object_points, (ground - object_pos[frame]) @ object_T[frame, :3, :3]]))

scene_xml = scene_dir / "g1_29dof_spherehand_w_object.xml"
constants = SimpleNamespace(
    ROBOT_URDF_FILE=str(scene_xml).replace("_w_object.xml", ".urdf"),  # the solver loads <urdf>_w_<object>.xml
    OBJECT_NAME="object",
    SCENE_XML_FILE=str(scene_xml),
    ROBOT_DOF=29,
    DEMO_JOINTS=joint_names,
    JOINTS_MAPPING=JOINTS_MAPPING,
    FOOT_STICKING_LINKS=[f"{side}_ankle_roll_sphere_{i}_link" for i in range(1, 5) for side in ("left", "right")],
    # holosoma's G1 bounds and costs (indices into the optimised [base 7 + 29 joints] vector).
    MANUAL_LB={"3": -1.0, "4": -1.0, "5": -1.0, "6": -1.0, "20": -0.3, "21": -0.1,
               "26": -0.1, "27": -0.1, "28": -0.05, "33": -0.1, "34": -0.1, "35": -0.05},
    MANUAL_UB={"3": 1.0, "4": 1.0, "5": 1.0, "6": 1.0, "20": 0.3, "21": 0.1,
               "26": 0.1, "27": 0.1, "28": 0.05, "33": 0.1, "34": 0.1, "35": 0.05},
    MANUAL_COST={"19": 0.2, "20": 0.2},
    NOMINAL_TRACKING_INDICES=np.arange(19),
)
foot_sticking = extract_foot_sticking_sequence_velocity(human_joints, joint_names, TOE_NAMES)
foot_sticking[0] = {key: False for key in foot_sticking[0]}  # keys L_Toe / R_Toe; as in holosoma's example
retargeter = InteractionMeshRetargeter(constants, object_urdf_path=None, activate_foot_sticking=True,
                                       activate_obj_non_penetration=True, activate_joint_limits=True)

# First-frame initial guess: pelvis above the human's, facing along the human's right-to-left hip direction, joints 0.
hips = human_joints[0, joint_names.index("Hips")]
left_dir = human_joints[0, joint_names.index("LeftLeg")] - human_joints[0, joint_names.index("RightLeg")]
yaw = np.arctan2(left_dir[1], left_dir[0]) - np.pi / 2  # robot +y points to its left
q_init = np.zeros(retargeter.nq)
q_init[:3] = [hips[0], hips[1], PELVIS_HEIGHT]
q_init[3:7] = Rotation.from_euler("z", yaw).as_quat(scalar_first=True)
q_init[-7:] = object_poses[0]

result_npz = sequence_dir / "omniretarget_qpos.npz"
retargeter.retarget_motion(
    human_joint_motions=human_joints * scale,
    object_poses=demo_object_poses,
    object_poses_augmented=object_poses,
    object_points_local_demo=[points * scale for points in points_local],
    object_points_local=points_local,
    foot_sticking_sequences=foot_sticking,
    q_a_init=q_init[retargeter.q_a_indices],
    original=True,
    dest_res_path=str(result_npz),
)
print(f"infeasible SQP steps recovered by fallback: {getattr(retargeter, 'infeasible_fallbacks', 0)}")
qpos = np.load(result_npz)["qpos"]
assert qpos.shape[0] == frames, (qpos.shape, frames)

# Zero-phase low-pass of the robot part: root position, root quaternion (sign-continuous), 29 joints.
b, a = butter(4, SMOOTHING_CUTOFF_HZ / (FPS / 2))
root_quat = qpos[:, 3:7].copy()
for frame in range(1, frames):
    if np.dot(root_quat[frame], root_quat[frame - 1]) < 0:
        root_quat[frame] = -root_quat[frame]
qpos[:, :3] = filtfilt(b, a, qpos[:, :3], axis=0)
qpos[:, 3:7] = filtfilt(b, a, root_quat, axis=0)
qpos[:, 7:36] = filtfilt(b, a, qpos[:, 7:36], axis=0)

# Robot motion in the official 43-joint layout (fingers 0), palm poses by FK of the G1 + Dex3 URDF.
mj_joint_names = [retargeter.robot_model.joint(j).name for j in range(1, 30)]
root_pos = qpos[:, :3]
root_wxyz = qpos[:, 3:7] / np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
joint_positions = np.zeros((frames, len(G1_JOINT_NAMES)))
for k, name in enumerate(mj_joint_names):
    joint_positions[:, G1_JOINT_NAMES.index(name)] = qpos[:, 7 + k]

urdf = yourdfpy.URDF.load(G1_DEX3_URDF, load_meshes=False, build_collision_scene_graph=False,
                          load_collision_meshes=False)
ee_pose = np.zeros((frames, len(EE_LINKS), 7))
for frame in range(frames):
    urdf.update_cfg(dict(zip(G1_JOINT_NAMES, joint_positions[frame])))
    world_T_root = trimesh.transformations.quaternion_matrix(root_wxyz[frame])
    world_T_root[:3, 3] = root_pos[frame]
    for k, link in enumerate(EE_LINKS):
        world_T_link = world_T_root @ urdf.get_transform(link)
        ee_pose[frame, k, :3] = world_T_link[:3, 3]
        ee_pose[frame, k, 3:] = trimesh.transformations.quaternion_from_matrix(world_T_link)

# Hand contact labels as soma_to_g1.py defines them: palm or a human fingertip within 2 cm of an object vertex.
contact = np.zeros((2, frames))
for side_index, prefix in enumerate(("Left", "Right")):
    tips = [joint_names.index(f"{prefix}Hand{finger}End") for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky")
            if f"{prefix}Hand{finger}End" in joint_names]
    for frame in range(frames):
        vertices_w = object_vertices @ object_T[frame, :3, :3].T + object_pos[frame]
        points = np.vstack([ee_pose[frame, side_index, :3], human_joints[frame, tips]])
        squared = ((vertices_w[:, None, :] - points[None, :, :]) ** 2).sum(-1)
        contact[side_index, frame] = float(squared.min() < CONTACT_THRESHOLD ** 2)

object_name = f"{sequence_id}_object"
safe_object_name = f"obj_{object_name}" if object_name[0].isdigit() else object_name  # soma_to_g1._usd_safe
motion = MotionData(
    sequence_id=sequence_id,
    robot_name="g1",
    motion_kind="single_robot",
    source_dataset="soma",
    raw_motion_file=str((folder / "soma_params.npz").resolve()),
    fps=FPS,
    coord_frame="robot_base_z_up",
    robot_joint_names=G1_JOINT_NAMES,
    robot_root_position=root_pos.tolist(),
    robot_root_wxyz=root_wxyz.tolist(),
    robot_joint_positions=joint_positions.tolist(),
    ee_link_names=EE_LINKS,
    ee_pose_w=ee_pose.tolist(),
    hand_sides=["left", "right"],
    hand_contact_active=contact.tolist(),
    object_name=object_name,
    safe_object_name=safe_object_name,
    object_body_names=["object"],
    safe_object_body_names=["object"],
    object_mesh_paths=[str((object_dir / "textured_mesh.obj").resolve())],
    object_urdf_paths=[str((object_dir / "textured_mesh.urdf").resolve())],
    object_mesh_radius=[float(np.linalg.norm(object_vertices - object_vertices.mean(0), axis=1).max())],  # as soma_to_g1
    object_articulation=[0.0] * frames,
    object_root_axis_angle=Rotation.from_matrix(object_T[:, :3, :3]).as_rotvec().tolist(),
    object_root_position=object_pos.tolist(),
    object_body_position=object_pos[:, None, :].tolist(),
    object_body_wxyz=object_wxyz[:, None, :].tolist(),
    source_kind="soma",
    source_payload=pickle.dumps({"retargeter": "omniretarget", "scale": scale, "object_ground_lift": lift}),
)
save_motion_parquet(motion, root_path=str(motion_root / "whole_body" / "soma"), file_name="data.parquet")
print(f"wrote {sequence_dir / 'robot_name=g1' / 'data.parquet'} (scale {scale:.3f}, object lift {lift:.4f} m)")
