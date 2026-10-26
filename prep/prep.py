"""Convert a LeRobot Track 2 dataset into the inputs of the official retargeter and FlashCHORD.

    python prep/prep.py <dataset_dir> <soma_dir> <stage_dir>

<dataset_dir>  a released tier, e.g. .../track_2/tier_1_multiview_caption
<soma_dir>     gets one retargeter input folder per sequence:
                   sequence_id=<seq>/soma_params.npz                        SOMA-X body parameters
                   sequence_id=<seq>/poses.npy                              (T, 4, 4) world_T_object, OpenCV world
                   sequence_id=<seq>/ground_plane.json                      copied unchanged
                   sequence_id=<seq>/reconstructed_mesh/output_aligned.glb  copied unchanged
<stage_dir>    gets <seq>_support.usda (copied unchanged). FlashCHORD looks for it in
               <motion_root>/whole_body/reconstructed_stage/, where <motion_root> is the
               --motion-root given to soma_to_g1.py.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation, Slerp


def read_column(table, name):
    """fixed_size_list<float32, N> column -> (T, N) array."""
    column = table.column(name).combine_chunks()
    return column.flatten().to_numpy().reshape(len(column), -1).astype(np.float32)


dataset_dir, soma_dir, stage_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
stage_dir.mkdir(parents=True, exist_ok=True)

# Joint order of observation.human.pose: feature names are "Hips.rx", "Hips.ry", "Hips.rz", ...
info = json.loads((dataset_dir / "meta" / "info.json").read_text())
joint_names = np.array([name[:-3] for name in info["features"]["observation.human.pose"]["names"][::3]])

for line in open(dataset_dir / "meta" / "episodes_metadata.jsonl"):
    episode = json.loads(line)
    sequence_id = episode["sequence_id"]
    table = pq.read_table(dataset_dir / "data" / "chunk-000" / f"episode_{episode['episode_index']:06d}.parquet")
    folder = soma_dir / f"sequence_id={sequence_id}"
    (folder / "reconstructed_mesh").mkdir(parents=True, exist_ok=True)

    np.savez(
        folder / "soma_params.npz",
        poses=read_column(table, "observation.human.pose").reshape(len(table), -1, 3),
        transl=read_column(table, "observation.human.translation"),
        identity_coeffs=read_column(table, "observation.human.identity_coeffs"),
        scale_params=read_column(table, "observation.human.scale_params"),
        joint_names=joint_names,
        identity_model_type="mhr",
        keep_root=False,
        absolute_pose=False,
        rotation_repr="rotvec",
        unit="meters",
        # read_soma.py requires this key but ignores it (the SOMA model owns the rest pose).
        joint_orient=np.tile(np.eye(3, dtype=np.float32), (len(joint_names) + 1, 1, 1)),
    )

    # Object pose rows are [x, y, z, qw, qx, qy, qz]. Invisible frames are all zeros: fill them by
    # interpolating the visible neighbours (the scorer skips these frames anyway).
    pose = read_column(table, "observation.object.pose").astype(np.float64)
    visible = table.column("observation.object.visible").to_numpy(zero_copy_only=False).astype(bool)
    if not visible.all():
        frames, known = np.arange(len(pose)), np.flatnonzero(visible)
        for axis in range(3):
            pose[:, axis] = np.interp(frames, known, pose[known, axis])
        slerp = Slerp(known, Rotation.from_quat(pose[known, 3:], scalar_first=True))
        pose[:, 3:] = slerp(np.clip(frames, known[0], known[-1])).as_quat(scalar_first=True)
    world_T_object = np.tile(np.eye(4), (len(pose), 1, 1))
    world_T_object[:, :3, :3] = Rotation.from_quat(pose[:, 3:], scalar_first=True).as_matrix()
    world_T_object[:, :3, 3] = pose[:, :3]
    np.save(folder / "poses.npy", world_T_object)

    shutil.copy2(dataset_dir / episode["ground_plane"], folder / "ground_plane.json")
    shutil.copy2(dataset_dir / episode["mesh"], folder / "reconstructed_mesh" / "output_aligned.glb")
    shutil.copy2(dataset_dir / episode["support_surface"], stage_dir / f"{sequence_id}_support.usda")
    print(f"{folder}  ({len(table)} frames, {(~visible).sum()} invisible)")
