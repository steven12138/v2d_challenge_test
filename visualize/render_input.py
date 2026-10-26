"""Render the retargeter's input (SOMA-X human mesh + object) to MP4, in the same G1 world as the
retargeted motion, so it can be compared side by side with visualize/render.py's reference.mp4.

    python visualize/render_input.py <soma_dir>/sequence_id=<seq> <out.mp4>

The human is placed by retarget/human_world.py (the transforms of soma_to_g1.py). The camera matches
visualize/render.sh.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "retarget"))
sys.path.insert(0, str(ROOT / "train" / "flash_chord" / "src"))

import newton
import numpy as np
import trimesh
import warp as wp

from flash_chord.visualization.viewer import ViewerConfig, configure_camera, create_viewer
from human_world import human_in_robot_world

folder, output = Path(sys.argv[1]), Path(sys.argv[2])
human = human_in_robot_world(folder)
human_vertices, object_poses = human["vertices"], human["object_poses"]
object_mesh = trimesh.load(folder / "reconstructed_mesh" / "output_aligned.glb", force="mesh")
object_vertices = np.asarray(object_mesh.vertices)

# Newton's headless GL viewer with only a ground plane in the model, recorded by FlashCHORD's MP4Viewer.
frames = len(human_vertices)
config = ViewerConfig(backend="mp4", output_path=str(output), num_frames=frames, video_fps=30.0,
                      width=1280, height=720, realtime=False,
                      camera_position=(3.0, -3.0, 1.8), camera_target=(0.0, 0.0, 0.7))
viewer = create_viewer(config)
builder = newton.ModelBuilder()
builder.add_ground_plane()
model = builder.finalize()
state = model.state()
viewer.set_model(model)
configure_camera(viewer, config)

human_faces = wp.array(human["faces"].reshape(-1).astype(np.int32), dtype=wp.int32)
object_faces = wp.array(np.asarray(object_mesh.faces).reshape(-1).astype(np.int32), dtype=wp.int32)
for frame in range(frames):
    pose = object_poses[frame]
    viewer.begin_frame(frame / 30.0)
    viewer.log_state(state)
    viewer.log_mesh("human", wp.array(human_vertices[frame].astype(np.float32), dtype=wp.vec3), human_faces,
                    color=(0.85, 0.65, 0.5))
    viewer.log_mesh("object", wp.array((object_vertices @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32),
                                       dtype=wp.vec3), object_faces, color=(0.55, 0.45, 0.3))
    viewer.end_frame()
viewer.close()
