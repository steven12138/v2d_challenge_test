"""Build the MuJoCo scenes with the object as a free body:
  g1_29dof_spherehand_w_object.xml  G1, 29 DoF with sphere hands (holosoma model), used by OmniRetarget
  g1_dex3_w_object.xml              the official G1 + Dex3 URDF (all 43 joints), used by retarget/eval_retarget.py
                                    and visualize/keyframes.py, so retargeters are judged on the trained robot

    python retarget/omniretarget/build_scene.py <object mesh (.obj/.glb)> <out_dir>

MuJoCo collides each mesh geom as its convex hull, which would fill the space under a table top. So the object is
split into convex parts with CoACD (as FlashCHORD does for its collision meshes), and every part becomes one geom
named object_part_<i>. The interaction-mesh retargeter finds object geoms by the name "object".
Also writes <out_dir>/object_part_<i>.obj.
"""

import sys
from pathlib import Path

import coacd
import mujoco
import numpy as np
import trimesh

ROBOT_XML = Path(__file__).resolve().parent / "models" / "g1" / "g1_29dof_spherehand.xml"
G1_DEX3_URDF = (Path(__file__).resolve().parents[1] / "robotic_grounding" / "robotic_grounding" / "assets" / "urdfs"
                / "g1" / "main_with_hand.urdf")

mesh_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

mesh = trimesh.load(mesh_path, force="mesh")
parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=0.05)
assets, geoms = [], []
for i, (vertices, faces) in enumerate(parts):
    trimesh.Trimesh(np.asarray(vertices), np.asarray(faces)).export(out_dir / f"object_part_{i}.obj")
    assets.append(f'    <mesh name="object_part_{i}" file="{(out_dir / f"object_part_{i}.obj").resolve()}"/>')
    geoms.append(f'      <geom name="object_part_{i}" type="mesh" mesh="object_part_{i}" contype="1" conaffinity="1" '
                 f'rgba="0.7 0.8 0.9 0.7"/>')

xml = ROBOT_XML.read_text()
xml = xml.replace('meshdir="meshes"', f'meshdir="{(ROBOT_XML.parent / "meshes").resolve()}"')
xml = xml.replace("  <asset>\n", "  <asset>\n" + "\n".join(assets) + "\n", 1)
body = "    <body name=\"object_link\">\n      <freejoint/>\n" \
       "      <inertial pos=\"0 0 0\" mass=\"1.0\" diaginertia=\"0.01 0.01 0.01\"/>\n" + "\n".join(geoms) + "\n    </body>\n"
xml = xml.replace("  </worldbody>", body + "  </worldbody>", 1)
(out_dir / "g1_29dof_spherehand_w_object.xml").write_text(xml)

# Official G1 + Dex3: the URDF is fixed-base, so add a free joint at the pelvis, a ground plane and the object.
spec = mujoco.MjSpec.from_file(str(G1_DEX3_URDF))
spec.meshdir = str(G1_DEX3_URDF.parents[2] / "meshes" / "g1")
spec.worldbody.first_body().add_freejoint()
spec.worldbody.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[10, 10, 0.1])
object_body = spec.worldbody.add_body(name="object_link")
object_body.add_freejoint()
for i in range(len(parts)):
    spec.add_mesh(name=f"object_part_{i}", file=str((out_dir / f"object_part_{i}.obj").resolve()))
    object_body.add_geom(name=f"object_part_{i}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"object_part_{i}",
                         rgba=[0.7, 0.8, 0.9, 0.6])
spec.compile()
(out_dir / "g1_dex3_w_object.xml").write_text(spec.to_xml())
print(f"{len(parts)} convex parts -> {out_dir}/g1_29dof_spherehand_w_object.xml, g1_dex3_w_object.xml")
