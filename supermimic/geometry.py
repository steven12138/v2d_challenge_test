"""Differentiable rotation helpers (rotation matrices; quaternions are xyzw as in FlashCHORD)."""

from __future__ import annotations

import jax.numpy as jnp


def quat_to_mat(q):
    """xyzw quaternion (..., 4) -> rotation matrix (..., 3, 3); same formula as FlashCHORD's 6D kernel."""
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = jnp.stack(
        [
            1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w),
            2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
            2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y),
        ],
        axis=-1,
    )
    return m.reshape(q.shape[:-1] + (3, 3))


def mat_to_6d(m):
    """First two columns, row-major flatten (FlashCHORD `_rotation_6d_component` order)."""
    return m[..., :, :2].reshape(m.shape[:-2] + (6,))


def six_d_to_mat(d):
    """Gram-Schmidt inverse of `mat_to_6d`."""
    cols = d.reshape(d.shape[:-1] + (3, 2))
    a, b = cols[..., 0], cols[..., 1]
    e1 = a / jnp.sqrt(jnp.sum(a * a, axis=-1, keepdims=True) + 1e-12)
    b = b - jnp.sum(e1 * b, axis=-1, keepdims=True) * e1
    e2 = b / jnp.sqrt(jnp.sum(b * b, axis=-1, keepdims=True) + 1e-12)
    e3 = jnp.cross(e1, e2)
    return jnp.stack([e1, e2, e3], axis=-1)


def mT(m):
    return jnp.swapaxes(m, -1, -2)


def mv(m, v):
    return jnp.einsum("...ij,...j->...i", m, v)


def mm(a, b):
    return jnp.einsum("...ij,...jk->...ik", a, b)


def rot_angle_sq(r_rel):
    """Squared geodesic angle of a relative rotation matrix, smooth at zero."""
    c = 0.5 * (jnp.trace(r_rel, axis1=-2, axis2=-1) - 1.0)
    s = 0.5 * jnp.stack(
        [r_rel[..., 2, 1] - r_rel[..., 1, 2], r_rel[..., 0, 2] - r_rel[..., 2, 0], r_rel[..., 1, 0] - r_rel[..., 0, 1]],
        axis=-1,
    )
    sn = jnp.sqrt(jnp.sum(s * s, axis=-1) + 1e-12)
    ang = jnp.arctan2(sn, c)
    return ang * ang


def geodesic_angle(ra, rb):
    """Angle between rotations ra, rb (matches 2*acos|<qa,qb>|)."""
    return jnp.sqrt(rot_angle_sq(mm(mT(ra), rb)) + 1e-12)


def log_map(r):
    """Rotation matrix -> rotation vector (shortest path)."""
    c = 0.5 * (jnp.trace(r, axis1=-2, axis2=-1) - 1.0)
    s = 0.5 * jnp.stack(
        [r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]], axis=-1
    )
    sn = jnp.sqrt(jnp.sum(s * s, axis=-1) + 1e-12)
    ang = jnp.arctan2(sn, c)
    return s * (ang / sn)[..., None]


def yaw_of(r):
    return jnp.arctan2(r[..., 1, 0], r[..., 0, 0])


def rot_z(yaw):
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    z, o = jnp.zeros_like(yaw), jnp.ones_like(yaw)
    return jnp.stack([c, -s, z, s, c, z, z, z, o], axis=-1).reshape(yaw.shape + (3, 3))
