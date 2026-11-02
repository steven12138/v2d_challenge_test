import os, sys, collections
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np, warp as wp, newton
from supermimic import env_io
with wp.ScopedDevice("cuda:0"):
    h = env_io.build(sys.argv[1], 1)
    m = h.env.model
    print("attrs:", [a for a in dir(m) if a.startswith("shape_")][:60])
    n = m.shape_count
    types = m.shape_type.numpy(); body = m.shape_body.numpy(); flags = m.shape_flags.numpy(); scale = m.shape_scale.numpy()
    print("shape_count", n, "body_count", m.body_count, "types", collections.Counter(types.tolist()))
    print("GeoType", {k: int(v) for k, v in vars(newton.GeoType).items() if not k.startswith("_") and isinstance(v, int)})
    print("ShapeFlags", {k: int(v) for k, v in vars(newton.ShapeFlags).items() if not k.startswith("_") and isinstance(v, int)})
    keys = getattr(m, "shape_key", None) or getattr(m, "shape_label", None)
    bkeys = getattr(m, "body_key", None) or getattr(m, "body_label", None)
    for i in range(n):
        src = m.shape_source[i] if hasattr(m, "shape_source") else None
        nv = None if src is None or not hasattr(src, "vertices") else len(src.vertices)
        print(i, keys[i] if keys else "", "type", types[i], "body", body[i], bkeys[body[i]] if (bkeys and body[i] >= 0) else "-", "flags", flags[i], "scale", np.round(scale[i], 3), "nverts", nv)
