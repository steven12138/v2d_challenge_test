"""Multi-view keyframe images of retargeted motions, to check penetration and contact by eye.

    python visualize/keyframes.py <scene xml> <out.png> <frame,frame,...> <label>=<motion data.parquet> ...

Renders the G1 (sphere hands) + object scene from retarget/omniretarget/build_scene.py with MuJoCo, offscreen.
Each frame becomes a block with one row per motion and three views around the object: front-left, back-right
and top-down. The object is translucent and penetrating contacts are drawn in red, so a hand inside the object
is visible.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

VIEWS = (("front-left", 45.0, -20.0), ("back-right", 225.0, -20.0), ("top", 90.0, -75.0))
WIDTH, HEIGHT = 480, 360

scene_xml, out_png, frame_list = Path(sys.argv[1]), Path(sys.argv[2]), [int(f) for f in sys.argv[3].split(",")]
motions = [arg.split("=", 1) for arg in sys.argv[4:]]

model = mujoco.MjModel.from_xml_path(str(scene_xml))
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
options = mujoco.MjvOption()
options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
model.vis.scale.contactwidth, model.vis.scale.contactheight = 0.06, 0.02
model.vis.rgba.contactpoint[:] = (1.0, 0.0, 0.0, 1.0)
body_joint_names = [model.joint(j).name for j in range(1, model.njnt - 1)]
object_body = model.body("object_link").id


def qpos_sequence(path):
    row = pq.read_table(path).to_pylist()[0]
    joints = dict(zip(row["robot_joint_names"], np.asarray(row["robot_joint_positions"]).T))
    return np.concatenate([np.asarray(row["robot_root_position"]), np.asarray(row["robot_root_wxyz"]),
                           np.stack([joints[name] for name in body_joint_names], 1),
                           np.asarray(row["object_body_position"])[:, 0], np.asarray(row["object_body_wxyz"])[:, 0]],
                          axis=1)


sequences = {label: qpos_sequence(path) for label, path in motions}
blocks = []
for frame in frame_list:
    rows = []
    for label, qpos in sequences.items():
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)  # contacts at zero margin: only real penetrations are drawn
        tiles = []
        for name, azimuth, elevation in VIEWS:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = 0.5 * (data.xpos[object_body] + data.xpos[1])  # between object and pelvis
            camera.distance, camera.azimuth, camera.elevation = 2.2, azimuth, elevation
            renderer.update_scene(data, camera, options)
            tile = Image.fromarray(renderer.render())
            ImageDraw.Draw(tile).text((8, 8), f"{label} | frame {frame} | {name}", fill=(255, 255, 0))
            tiles.append(np.asarray(tile))
        rows.append(np.concatenate(tiles, axis=1))
    blocks.append(np.concatenate(rows, axis=0))
out_png.parent.mkdir(parents=True, exist_ok=True)
Image.fromarray(np.concatenate(blocks, axis=0)).save(out_png)
print(f"wrote {out_png}")
