"""Device FIFO replay buffer: one ring slot per environment step, all worlds side by side."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from supermimic.geometry import quat_to_mat

RAW_DIM = 132  # pelvis(3+4+3+3) wrists(2*(3+4+3)) object(3+4+3+3) q(43) qd(43)


def pack_raw(k, body_q, body_qd, joint_q, joint_qd):
    pel, obj, wr = k["pelvis"], k["obj"], k["wrist"]
    W = body_q.shape[0]
    return jnp.concatenate(
        [
            body_q[:, pel, :7], body_qd[:, pel, :6],
            body_q[:, wr, :3].reshape(W, 6), body_q[:, wr, 3:7].reshape(W, 8), body_qd[:, wr, :3].reshape(W, 6),
            body_q[:, obj, :7], body_qd[:, obj, :6],
            joint_q[:, k["q_ids"]], joint_qd[:, k["dof_ids"]],
        ],
        axis=-1,
    )


def unpack_raw(x):
    B = x.shape[0]
    return dict(
        pp=x[:, 0:3], pR=quat_to_mat(x[:, 3:7]), pv=x[:, 7:10], pw=x[:, 10:13],
        hp=x[:, 13:19].reshape(B, 2, 3), hR=quat_to_mat(x[:, 19:27].reshape(B, 2, 4)), hv=x[:, 27:33].reshape(B, 2, 3),
        op=x[:, 33:36], oR=quat_to_mat(x[:, 36:40]), ov=x[:, 40:43], ow=x[:, 43:46],
        q=x[:, 46:89], qd=x[:, 89:132],
    )


FIELDS = {
    # name: (trailing shape, dtype)
    "s": ((RAW_DIM,), jnp.float32),
    "sn": ((RAW_DIM,), jnp.float32),
    "frame": ((), jnp.int32),
    "estep": ((), jnp.int32),
    "age": ((), jnp.int32),
    "start": ((), jnp.int32),
    "voc": ((), jnp.float32),
    "lags": ((5,), jnp.int32),
    "action": ((43,), jnp.float32),
    "target": ((43,), jnp.float32),
    "decoded": ((29,), jnp.float32),
    "cur": ((93,), jnp.float32),
    "terms": ((12,), jnp.float32),
    "term": ((), jnp.bool_),
    "trunc": ((), jnp.bool_),
    "c_act": ((16,), jnp.float32),
    "c_pos": ((16, 3), jnp.float32),
    "c_dir": ((16, 3), jnp.float32),
}


def create(capacity: int, world_count: int) -> dict:
    buf = {name: jnp.zeros((capacity, world_count) + shape, dtype) for name, (shape, dtype) in FIELDS.items()}
    buf["g"] = jnp.full((capacity,), -1, jnp.int32)  # global step stored in each ring slot
    return buf


@partial(jax.jit, donate_argnums=(0,))
def write(buf, slot, g, row):
    out = {name: buf[name].at[slot].set(row[name].astype(buf[name].dtype)) for name in FIELDS}
    out["g"] = buf["g"].at[slot].set(g)
    return out


def gather(buf, g, w, capacity):
    """Rows at global steps g (B,) for worlds w (B,)."""
    slot = jnp.mod(g, capacity)
    return {name: buf[name][slot, w] for name in FIELDS}
