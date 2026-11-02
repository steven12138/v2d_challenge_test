"""Hand report of a recorded rollout (record_rollout.py): close-up contact sheet + text summary of the interaction.

Run with the render venv: python supermimic/render_hands.py <rollout.npz> <out_prefix> <title>
Writes <out_prefix>.png (12 close-ups through the interaction) and <out_prefix>.txt.
Env FRAMES=lo,hi overrides the frame window; VIEW=body frames the whole robot instead of the hand.
"""

from __future__ import annotations

import os
import sys

from render_rollout import angle_deg, look_at, tf  # sets up the headless pyrender imports

import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageDraw, ImageFont

TILE_W, TILE_H, COLS, ROWS = 560, 420, 4, 3


def interaction_window(ref_p):
    """Frames from just before the reference object starts moving to just after it stops."""
    speed = np.linalg.norm(np.diff(ref_p, axis=0), axis=1) * 50.0
    moving = np.nonzero(np.convolve(speed, np.ones(5) / 5, mode="same") > 0.05)[0]
    if len(moving) == 0:
        return 0, len(ref_p) - 1
    return max(int(moving[0]) - 20, 0), min(int(moving[-1]) + 20, len(ref_p) - 1)


def report(d, lo, hi, title):
    body_q, ref_body_q = d["body_q"], d["ref_body_q"]
    obj, pelvis = int(d["object_body"]), int(d["pelvis_body"])
    wrists = d["wrist_bodies"]  # [right, left]
    ref_p, ref_q, contact = d["ref_obj_pos"], d["ref_obj_quat_xyzw"], d["contact"]
    F = len(ref_p)
    sim = body_q[1:]  # state after the step for frame f
    err = np.linalg.norm(sim[:, obj, :3] - ref_p, axis=1)
    ang = np.array([angle_deg(sim[f, obj, 3:7], ref_q[f]) for f in range(F)])
    wrist_err = np.linalg.norm(sim[:, wrists, :3] - ref_body_q[:, wrists, :3], axis=-1)
    left, right = contact[:, :8].sum(1), contact[:, 8:].sum(1)
    away = np.convolve(err > 0.10, np.ones(10), mode="valid") >= 10
    lost = np.nonzero(away)[0]
    lines = [title, f"interaction frames {lo}-{hi} of {F - 1}",
             "frame refz  objz  err_cm  deg  contacts(L R)  wristerr_cm(L R)  pelvisz"]
    for f in np.linspace(lo, hi, 13).astype(int):
        lines.append(f"{f:5d} {ref_p[f, 2]:.2f}  {sim[f, obj, 2]:.2f}  {100 * err[f]:6.1f} {ang[f]:4.0f}   "
                     f"{int(left[f]):2d} {int(right[f]):2d}          {100 * wrist_err[f, 1]:5.1f} {100 * wrist_err[f, 0]:5.1f}      "
                     f"{sim[f, pelvis, 2]:.2f}")
    w = np.arange(lo, hi + 1)
    lines += [
        f"whole episode: mean err {100 * err.mean():.1f} cm, mean angle {ang.mean():.0f} deg, min pelvis z {sim[:, pelvis, 2].min():.2f}",
        f"interaction: mean err {100 * err[w].mean():.1f} cm; frames <3 cm {np.mean(err[w] < 0.03):.2f}, <10 cm {np.mean(err[w] < 0.10):.2f}, <20 cm {np.mean(err[w] < 0.20):.2f}",
        f"interaction: frames touching L {np.mean(left[w] > 0):.2f} R {np.mean(right[w] > 0):.2f}; mean links L {left[w].mean():.1f} R {right[w].mean():.1f}",
        f"object first stays >10 cm off from frame {int(lost[0]) if len(lost) else 'never'}; peak object z {sim[:, obj, 2].max():.2f} (reference {ref_p[:, 2].max():.2f})",
    ]
    return "\n".join(lines), err, ang, left, right


def main(npz: str, prefix: str, title: str) -> None:
    d = np.load(npz)
    body_q, ref_body_q = d["body_q"], d["ref_body_q"]
    obj, pelvis = int(d["object_body"]), int(d["pelvis_body"])
    wrists = d["wrist_bodies"]
    ref_p, ref_q = d["ref_obj_pos"], d["ref_obj_quat_xyzw"]
    lo, hi = interaction_window(ref_p)
    if os.environ.get("FRAMES"):
        lo, hi = (int(x) for x in os.environ["FRAMES"].split(","))
    body_view = os.environ.get("VIEW") == "body"
    text, err, ang, left, right = report(d, lo, hi, title)
    with open(prefix + ".txt", "w") as fh:
        fh.write(text + "\n")
    print(text, flush=True)

    scene = pyrender.Scene(bg_color=[0.93, 0.94, 0.96, 1.0], ambient_light=[0.45, 0.45, 0.45])
    robot_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.35, 0.37, 0.40, 1.0], metallicFactor=0.2, roughnessFactor=0.6)
    obj_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.95, 0.55, 0.15, 1.0], roughnessFactor=0.7)
    ghost_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.2, 0.8, 0.3, 0.35], alphaMode="BLEND")
    ref_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.2, 0.45, 0.95, 0.25], alphaMode="BLEND", roughnessFactor=0.7)
    nodes, ref_nodes, obj_mesh = [], [], None
    for key in d.files:
        if not key.endswith("_v"):
            continue
        stem = key[:-2]
        body = int(d[stem + "_body"])
        tm = trimesh.Trimesh(d[stem + "_v"], d[stem + "_f"], process=False)
        local = tf(d[stem + "_tf"][:3], d[stem + "_tf"][3:7])
        if body == obj:
            obj_mesh = tm
        nodes.append((scene.add(pyrender.Mesh.from_trimesh(tm, material=obj_mat if body == obj else robot_mat)), body, local))
        if body != obj:
            ref_nodes.append((scene.add(pyrender.Mesh.from_trimesh(tm, material=ref_mat)), body, local))
    ghost = scene.add(pyrender.Mesh.from_trimesh(obj_mesh, material=ghost_mat))
    cam_node = scene.add(pyrender.PerspectiveCamera(yfov=np.radians(50)))
    key_light = scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0))
    renderer = pyrender.OffscreenRenderer(TILE_W, TILE_H)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
    except OSError:
        font = ImageFont.load_default()

    sheet = Image.new("RGB", (TILE_W * COLS, TILE_H * ROWS))
    for i, f in enumerate(np.linspace(lo, hi, COLS * ROWS).astype(int)):
        q = body_q[f + 1]
        for node, body, local in nodes:
            scene.set_pose(node, tf(q[body, :3], q[body, 3:7]) @ local)
        for node, body, local in ref_nodes:
            scene.set_pose(node, tf(ref_body_q[f, body, :3], ref_body_q[f, body, 3:7]) @ local)
        scene.set_pose(ghost, tf(ref_p[f], ref_q[f]))
        # look at the hand nearest the reference object, from the side of the robot
        hand = q[wrists[np.argmin(np.linalg.norm(q[wrists, :3] - ref_p[f], axis=1))], :3]
        target = 0.5 * (hand + ref_p[f])
        out = target - q[pelvis, :3]
        out[2] = 0.0
        out = out / max(np.linalg.norm(out), 1e-3)
        side = np.array([-out[1], out[0], 0.0])
        eye = target + 0.65 * side + 0.45 * out + np.array([0.0, 0.0, 0.40])
        if body_view:
            target = q[pelvis, :3] + np.array([0.0, 0.0, -0.15])
            eye = target + 1.9 * side + 0.9 * out + np.array([0.0, 0.0, 0.55])
        scene.set_pose(cam_node, look_at(eye, target))
        scene.set_pose(key_light, look_at(eye + np.array([0.0, 0.0, 1.5]), target))
        color, _ = renderer.render(scene)
        img = Image.fromarray(color)
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, TILE_W, 24), fill=(20, 22, 26))
        draw.text((6, 3), f"f{f} err {100 * err[f]:.0f}cm/{ang[f]:.0f}deg contacts L{int(left[f])} R{int(right[f])}",
                  fill=(240, 240, 240), font=font)
        sheet.paste(img, ((i % COLS) * TILE_W, (i // COLS) * TILE_H))
    renderer.delete()
    sheet.save(prefix + ".png")
    print(f"wrote {prefix}.png", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
