"""Deterministic ensemble world model f_w(s, a, xi) with contact and force-closure heads (paper Sec. 3.2)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from supermimic import core
from supermimic.buffer import unpack_raw


def mlp_init(key, sizes):
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        last = i == len(sizes) - 2
        scale = 0.01 if last else jnp.sqrt(2.0 / n_in)
        params.append(
            dict(
                w=jax.random.normal(keys[i], (n_in, n_out)) * scale,
                b=jnp.zeros((n_out,)),
                g=jnp.ones((n_out,)),
                beta=jnp.zeros((n_out,)),
            )
        )
    return params


def mlp_apply(params, x):
    for i, layer in enumerate(params):
        x = x @ layer["w"] + layer["b"]
        if i < len(params) - 1:
            mu = jnp.mean(x, -1, keepdims=True)
            var = jnp.var(x, -1, keepdims=True)
            x = (x - mu) * jax.lax.rsqrt(var + 1e-5) * layer["g"] + layer["beta"]
            x = jax.nn.silu(x)
    return x


def init(key, members, hidden, layers):
    sizes = [core.WM_IN_DIM] + [hidden] * layers + [core.WM_OUT_DIM]
    return jax.vmap(lambda k: mlp_init(k, sizes))(jax.random.split(key, members))


def normalizer_from(x, floor=1e-3):
    mean = jnp.mean(x, 0)
    std = jnp.maximum(jnp.std(x, 0), floor)
    return dict(mean=mean, std=std)


def step(params, k, norm, S, target, prev_target, lag1h, voc, frame):
    """One imagined control step for a single ensemble member."""
    x = core.wm_inputs(k, S, target, prev_target, lag1h, voc, frame)
    x = (x - norm["in"]["mean"]) / norm["in"]["std"]
    y = mlp_apply(params, x)
    D = core.DELTA_DIM
    delta = y[:, :D] * norm["out"]["std"] + norm["out"]["mean"]
    S_next = core.decode_delta(S, delta)
    B = y.shape[0]
    c = y[:, D : D + core.N_SLOTS * 7].reshape(B, core.N_SLOTS, 7)
    contact = dict(logit=c[..., 0], pos=c[..., 1:4] * 0.3, dir=c[..., 4:7])
    fc_logit = y[:, -1]
    return S_next, contact, fc_logit, y[:, :D]


def window_loss(params, k, norm, rows):
    """k-step autoregressive loss on logged actions (Eq. 1). rows: dict of (K, B, ...)."""
    K = rows["s"].shape[0]
    S = unpack_raw(rows["s"][0])
    total = 0.0
    parts = dict(state=0.0, contact=0.0, fc=0.0)
    for j in range(K):
        lag1h = core.lag_onehot(rows["lags"][j])
        prev = jnp.where((rows["estep"][j] > 0)[:, None], rows["prev_target"][j], rows["target"][j])
        S_hat, contact, fc_logit, y_norm = step(params, k, norm, S, rows["target"][j], prev, lag1h, rows["voc"][j], rows["frame"][j])
        S_true = unpack_raw(rows["sn"][j])
        target_delta = core.encode_delta(S, S_true)
        target_norm = (target_delta - norm["out"]["mean"]) / norm["out"]["std"]
        m = rows["valid"][j]
        l_state = jnp.sum(jnp.mean(k["delta_w"] * jnp.square(y_norm - target_norm), -1) * m) / jnp.maximum(jnp.sum(m), 1.0)
        act = rows["c_act"][j]
        bce = optax.sigmoid_binary_cross_entropy(contact["logit"], act)
        l_c = jnp.mean(bce, -1)
        l_c += jnp.sum(act[..., None] * jnp.square(contact["pos"] - rows["c_pos"][j]) / 0.05**2, (-1, -2)) / core.N_SLOTS
        l_c += jnp.sum(act[..., None] * jnp.square(contact["dir"] - rows["c_dir"][j]), (-1, -2)) / core.N_SLOTS
        l_c = jnp.sum(l_c * m) / jnp.maximum(jnp.sum(m), 1.0)
        l_fc = jnp.sum(jnp.square(jax.nn.sigmoid(fc_logit) - rows["terms"][j][:, 11]) * m) / jnp.maximum(jnp.sum(m), 1.0)
        total = total + l_state + 0.1 * l_c + l_fc
        parts["state"] += l_state / K
        parts["contact"] += l_c / K
        parts["fc"] += l_fc / K
        S = S_hat
    return total / K, parts


def make_optimizer(lr):
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
