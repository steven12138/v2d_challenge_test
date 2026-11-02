"""Render a recorded rollout (record_rollout.py) to MP4 with off-screen EGL pyrender.

Robot: grey visual meshes. Retargeted reference robot: translucent blue.
Object: orange (simulated). Reference object: translucent green ghost.
Camera follows the pelvis at a fixed 3/4 view. The overlay shows frame and object position / angle error.

usage (temporary env): uv run --no-project --with pyrender --with pyopengl==3.1.0 --with trimesh --with numpy \
    --with imageio --with imageio-ffmpeg --with pillow python render_rollout.py <rollout.npz> <out.mp4> <title>
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
# EGL ignores CUDA_VISIBLE_DEVICES and pyrender defaults to device 0; EGL devices 0-3 are the physical GPUs 0-3 on the server
os.environ.setdefault("EGL_DEVICE_ID", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])

import types

# pyrender imports its pyglet window viewer at package import; on a headless host pyglet cannot load
# (no display / GLU). Off-screen EGL rendering never touches it, so a stub satisfies the import.
_pyglet = types.ModuleType("pyglet")
_pyglet.options = {}
_pyglet.clock = types.ModuleType("pyglet.clock")
_pyglet.window = types.ModuleType("pyglet.window")
_pyglet.window.Window = object
sys.modules.update({"pyglet": _pyglet, "pyglet.clock": _pyglet.clock, "pyglet.window": _pyglet.window})

import imageio.v2 as imageio
import numpy as np

np.infty = np.inf  # pyrender still uses the alias NumPy 2 removed

import pyrender
import trimesh
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720


def quat_xyzw_to_mat(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def tf(p, q):
    T = np.eye(4)
    T[:3, :3] = quat_xyzw_to_mat(q)
    T[:3, 3] = p
    return T


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([s, u, -f])
    T[:3, 3] = eye
    return T


def angle_deg(qa, qb):
    d = abs(float(np.dot(qa / np.linalg.norm(qa), qb / np.linalg.norm(qb))))
    return float(np.degrees(2 * np.arccos(min(1.0, d))))


def main(npz: str, out: str, title: str) -> None:
    d = np.load(npz)
    body_q = d["body_q"]
    ref_body_q = d["ref_body_q"] if "ref_body_q" in d.files else None
    obj_body = int(d["object_body"])
    pelvis = int(d["pelvis_body"])
    ref_p, ref_q = d["ref_obj_pos"], d["ref_obj_quat_xyzw"]
    steps = body_q.shape[0] - 1

    scene = pyrender.Scene(bg_color=[0.93, 0.94, 0.96, 1.0], ambient_light=[0.35, 0.35, 0.35])
    floor = trimesh.creation.box((40.0, 40.0, 0.02))
    floor.apply_translation((0, 0, -0.01))
    checker = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.82, 0.83, 0.85, 1.0], roughnessFactor=0.9)
    scene.add(pyrender.Mesh.from_trimesh(floor, material=checker, smooth=False))
    for x in np.arange(-20, 20.01, 1.0):  # 1 m grid lines for scale / motion cues
        for line in (trimesh.creation.box((0.01, 40.0, 0.002)), trimesh.creation.box((40.0, 0.01, 0.002))):
            line.apply_translation((x, 0, 0.001) if line.extents[0] < 1 else (0, x, 0.001))
            scene.add(pyrender.Mesh.from_trimesh(line, material=pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.65, 0.66, 0.70, 1.0])))

    robot_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.35, 0.37, 0.40, 1.0], metallicFactor=0.2,
                                                   roughnessFactor=0.6)
    obj_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.95, 0.55, 0.15, 1.0], roughnessFactor=0.7)
    ghost_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.2, 0.8, 0.3, 0.35], alphaMode="BLEND")
    ref_robot_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.2, 0.45, 0.95, 0.28],
                                                       alphaMode="BLEND", roughnessFactor=0.7)
    nodes = []
    ref_nodes = []
    obj_mesh = None
    for key in d.files:
        if not key.endswith("_v"):
            continue
        stem = key[:-2]
        body = int(d[stem + "_body"])
        tm = trimesh.Trimesh(d[stem + "_v"], d[stem + "_f"], process=False)
        local = tf(d[stem + "_tf"][:3], d[stem + "_tf"][3:7])
        if body == obj_body:
            obj_mesh = tm
            mesh = pyrender.Mesh.from_trimesh(tm, material=obj_mat)
        else:
            mesh = pyrender.Mesh.from_trimesh(tm, material=robot_mat)
        nodes.append((scene.add(mesh), body, local))
        if ref_body_q is not None and body != obj_body:
            ref_mesh = pyrender.Mesh.from_trimesh(tm, material=ref_robot_mat)
            ref_nodes.append((scene.add(ref_mesh), body, local))
    ghost_node = scene.add(pyrender.Mesh.from_trimesh(obj_mesh, material=ghost_mat)) if obj_mesh is not None else None

    cam = pyrender.PerspectiveCamera(yfov=np.radians(50))
    cam_node = scene.add(cam)
    key_light = scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0))
    fill = scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.5))
    renderer = pyrender.OffscreenRenderer(W, H)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 26)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 19)
    except OSError:
        font = ImageFont.load_default()
        small_font = font

    # Camera: fixed 3/4 view direction, target follows a smoothed midpoint of pelvis and object.
    p0 = body_q[1, pelvis, :3]
    o0 = body_q[1, obj_body, :3]
    view_dir = np.array([1.0, -1.6, 0.0])
    side = np.cross(o0 - p0, [0, 0, 1])
    if np.linalg.norm(side) > 1e-3:
        view_dir = side / np.linalg.norm(side) * 1.6 + (o0 - p0) / max(np.linalg.norm(o0 - p0), 1e-3) * -0.8
    view_dir = view_dir / np.linalg.norm(view_dir)
    target = 0.5 * (p0 + o0)
    writer = imageio.get_writer(out, fps=float(d["fps"]), codec="libx264", quality=8, macro_block_size=8)
    errs = []
    for t in range(1, steps + 1):
        q = body_q[t]
        for node, body, local in nodes:
            scene.set_pose(node, tf(q[body, :3], q[body, 3:7]) @ local)
        f = min(t - 1, len(ref_p) - 1)
        if ref_body_q is not None:
            ref_robot_q = ref_body_q[f]
            for node, body, local in ref_nodes:
                scene.set_pose(node, tf(ref_robot_q[body, :3], ref_robot_q[body, 3:7]) @ local)
        if ghost_node is not None:
            scene.set_pose(ghost_node, tf(ref_p[f], ref_q[f]))
        mid = 0.5 * (q[pelvis, :3] + q[obj_body, :3])
        target = 0.94 * target + 0.06 * np.array([mid[0], mid[1], 0.6])
        eye = target + view_dir * 3.6 + np.array([0, 0, 1.4])
        scene.set_pose(cam_node, look_at(eye, target))
        scene.set_pose(key_light, look_at(eye + np.array([0, 0, 3.0]), target))
        scene.set_pose(fill, look_at(target + np.array([-2.0, 2.0, 2.0]), target))
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
        e = float(np.linalg.norm(q[obj_body, :3] - ref_p[f]))
        a = angle_deg(q[obj_body, 3:7], ref_q[f])
        errs.append(e)
        img = Image.fromarray(color)
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, W, 69), fill=(20, 22, 26))
        draw.text((14, 3), f"{title}   frame {f:4d}/{len(ref_p) - 1}", fill=(240, 240, 240), font=font)
        draw.text((14, 37), f"object err {100 * e:5.1f} cm / {a:5.1f} deg   mean so far {100 * np.mean(errs):4.1f} cm",
                  fill=(240, 240, 240), font=small_font)
        legend = "grey: simulated G1    blue ghost: reference G1    orange: simulated object    green: reference object"
        draw.text((14, H - 29), legend, fill=(30, 30, 30), font=small_font)
        writer.append_data(np.asarray(img))
    writer.close()
    renderer.delete()
    print(f"wrote {out}: {steps} frames, mean object error {100 * np.mean(errs):.2f} cm", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
