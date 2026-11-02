"""Short-horizon actor-critic through the frozen world model (paper Sec. 3.3, Eq. 2-3)."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import optax
from jax.ad_checkpoint import checkpoint_name

from flash_chord.training.flash_sac.network import project_unit_parameters

from supermimic import core, world_model
from supermimic.buffer import unpack_raw

N_EXTRA = len(core.EXTRA_NAMES)
OBJECT_EXTRAS = jnp.array([n in ("obj_ori", "obj_fine", "obj_vel", "obj_coarse", "obj_keypoints", "obj_keypoints_fine") for n in core.EXTRA_NAMES])


def critic_init(key, hidden):
    return world_model.mlp_init(key, [core.CRITIC_IN_DIM, hidden, hidden, hidden, 1])


def critic_apply(params, x):
    return world_model.mlp_apply(params, x)[..., 0]


def make_fns(k, cfg, actor_module, dec_params, wm_norm, critic_norm, weights, extra_weights, trainable=None):
    """Build jitted imagination / update functions closing over task constants."""
    F = k["F"]
    latent = k["latent"]
    M = cfg.wm_members

    def policy_pre(actor_params, batch_stats, obs):
        return actor_module.apply({"params": actor_params, "batch_stats": batch_stats}, obs, training=False).mean

    def policy_mean(actor_params, batch_stats, obs):
        return jnp.tanh(policy_pre(actor_params, batch_stats, obs)) * 10.0

    def critic_value(critic_params, feats):
        return critic_apply(critic_params, (feats - critic_norm["mean"]) / critic_norm["std"])

    def ff_at(ff, frame):
        """Feed-forward table: pre-tanh action offset at `frame`, linear between knots."""
        x = jnp.clip(frame, 0, F - 1) / cfg.ff_knot_frames
        i = jnp.floor(x).astype(jnp.int32)
        w = (x - i)[..., None]
        return ff[i] * (1.0 - w) + ff[jnp.minimum(i + 1, ff.shape[0] - 1)] * w

    def policy_ff(actor_params, batch_stats, ff, obs, frame):
        return jnp.tanh(policy_pre(actor_params, batch_stats, obs) + ff_at(ff, frame)) * 10.0

    def rollout(actor_params, batch_stats, wm_params, critic_params, seeds, horizon, voc_now, ff=None):
        """Imagine `horizon` steps for one ensemble member; returns objective and the trajectory."""
        S0 = unpack_raw(seeds["s"])
        lag1h = core.lag_onehot(seeds["lags"])
        B = seeds["s"].shape[0]
        # VOC of the replay row each rollout starts from: always data-supported for the world model
        # (the curriculum value itself may not be in the buffer yet right after a change).
        voc = seeds["voc"]
        free = None
        if cfg.policy_termination:
            t = k["term"]
            _, _, he, ho, oe, oo = core.tracking_errors(k, S0, seeds["frame"])
            free = ((jnp.max(he, -1) > t["palm_position"]) | (jnp.max(ho, -1) > t["palm_orientation"])
                    | (oe > t["object_position"]) | (oo > t["object_orientation"])).astype(jnp.float32)

        def body(carry, _):
            S, hist9, last_dec, tgt_hist, prev_a, prev_tgt, first, frame, estep, c, alive = carry
            obs = core.observation(k, S, frame, tgt_hist)
            pre = policy_pre(actor_params, batch_stats, obs)
            if ff is not None:
                pre = pre + ff_at(ff, frame)
            a = jnp.tanh(pre) * 10.0
            # saturation guard: a saturated tanh dimension has ~zero gradient and gets stuck at +-10
            sat = jnp.sum(jnp.square(jax.nn.relu(jnp.abs(pre) - cfg.sat_threshold)), -1)
            cur = core.sonic_current(k, S, last_dec)
            hist10 = jnp.concatenate([hist9, cur[:, None]], axis=1)
            dec = core.sonic_decode(dec_params, core.sonic_decoder_input(latent[jnp.clip(frame, 0, F - 1)], hist10))
            if cfg.sonic_stop_grad:
                dec = checkpoint_name(jax.lax.stop_gradient(dec), "sonic")
            tgt = core.processed_targets(k, dec, a, frame)
            prev_eff = jnp.where(first[:, None], tgt, prev_tgt)
            S1, contact, fc_logit, _ = world_model.step(wm_params, k, wm_norm, S, tgt, prev_eff, lag1h, voc, frame)
            d = core.termination_soft(k, S1, frame, estep, tau_frac=cfg.term_tau_frac, free=free)
            fc_an = core.force_closure(k, frame, jax.nn.sigmoid(contact["logit"]), contact["pos"], contact["dir"], sharp=0.005)
            fc = cfg.fc_head_mix * jax.nn.sigmoid(fc_logit) + (1 - cfg.fc_head_mix) * fc_an
            terms = core.objective_terms(k, S1, frame, seeds["start"], a, prev_a, fc, d)
            extra = core.extra_terms(k, S1, frame, jax.nn.sigmoid(contact["logit"]), tgt, a, cfg.hio_std)
            if cfg.obj_contact_gate > 0:
                # Omnigrasp: while the reference grasps, object tracking only pays when the hands touch the
                # object, so an object carried by the VOC (or lucky physics) earns little on its own
                grasping = (jnp.sum(k["grasp_lab"][jnp.clip(frame, 0, F - 1)], -1) > 0).astype(jnp.float32)
                gate = 1.0 - cfg.obj_contact_gate * grasping * (1.0 - extra[:, 3])
                terms = terms.at[:, 7].multiply(gate)
                extra = extra * jnp.where(OBJECT_EXTRAS, gate[:, None], 1.0)
            r = 0.02 * (terms @ weights + extra @ extra_weights)
            trunc = (frame >= F - 1).astype(jnp.float32)
            feat = core.critic_features(k, S, frame, estep, voc, lag1h)
            new_tgt_hist = jnp.concatenate([tgt_hist[:, 1:], tgt[:, None]], axis=1)
            new = (
                S1, hist10[:, 1:], dec, new_tgt_hist, a, tgt, jnp.zeros_like(first),
                jnp.minimum(frame + 1, F - 1), estep + 1, c * (1.0 - d), alive * (1.0 - trunc),
            )
            d_hard = core.termination_hard(k, S1, frame, estep).astype(jnp.float32)
            out = dict(r=r, d=d, d_hard=d_hard, trunc=trunc, c=c, alive=alive, feat=feat, terms=terms, extra=extra, sat=sat)
            return new, out

        carry0 = (
            S0, seeds["hist9"], seeds["last_dec"], seeds["tgt_hist"], seeds["prev_a"], seeds["prev_tgt"],
            seeds["first"], seeds["frame"], seeds["estep"], jnp.ones((B,)), jnp.ones((B,)),
        )
        step = body
        if cfg.imag_remat:
            keep = jax.checkpoint_policies.save_only_these_names("sonic") if cfg.sonic_stop_grad else None
            step = jax.checkpoint(body, policy=keep)
        carry, traj = jax.lax.scan(step, carry0, None, length=horizon)
        S_H, frame_H, estep_H, c_H, alive_H = carry[0], carry[7], carry[8], carry[9], carry[10]
        feat_H = core.critic_features(k, S_H, frame_H, estep_H, voc, lag1h)
        v_H = critic_value(jax.lax.stop_gradient(critic_params), feat_H)
        disc = cfg.gamma ** jnp.arange(horizon)[:, None]
        J = jnp.sum(disc * traj["c"] * traj["alive"] * traj["r"], 0) + cfg.gamma**horizon * c_H * alive_H * v_H
        traj["feat_H"] = feat_H
        return J, traj

    def grouped(fn, wm_params, seeds, *args):
        """Split the batch across ensemble members (one member per imagined rollout)."""
        seeds_g = jax.tree.map(lambda x: x.reshape((M, x.shape[0] // M) + x.shape[1:]), seeds)
        return jax.vmap(fn, in_axes=(0, 0) + (None,) * len(args))(wm_params, seeds_g, *args)

    @partial(jax.jit, static_argnames=("horizon",))
    def actor_update(actor_params, actor_opt, batch_stats, wm_params, critic_params, seeds, voc_now, anchor_params, horizon):
        # trust region: the world model is only good near the data of the start policy, so a gradient that leaves it exploits model errors
        obs0 = core.observation(k, unpack_raw(seeds["s"]), seeds["frame"], seeds["tgt_hist"])
        anchor_action = jax.lax.stop_gradient(policy_mean(anchor_params, batch_stats, obs0))

        def loss_fn(p):
            def member(wm_p, sd, p_, cp):
                return rollout(p_, batch_stats, wm_p, cp, sd, horizon, voc_now)

            J, traj = grouped(member, wm_params, seeds, p, critic_params)
            drift = jnp.mean(jnp.square(policy_mean(p, batch_stats, obs0) - anchor_action)) / 100.0
            return -jnp.mean(J) + cfg.sat_penalty * jnp.mean(traj["sat"]) + cfg.w_anchor * drift, (J, traj)

        (loss, (J, traj)), grads = jax.value_and_grad(loss_fn, has_aux=True)(actor_params)
        if trainable is not None:  # external residual: the base policy stays frozen and the unit projections would mix the branches
            grads = jax.tree.map(lambda g, m: g * m, grads, trainable)
        gnorm = optax.global_norm(grads)
        updates, actor_opt = actor_tx.update(grads, actor_opt, actor_params)
        actor_params = optax.apply_updates(actor_params, updates)
        if trainable is None:
            actor_params = project_unit_parameters(actor_params)
        return actor_params, actor_opt, jax.lax.stop_gradient(traj), imag_stats(loss, gnorm, J, traj)

    def imag_stats(loss, gnorm, J, traj):
        # traj leaves are (M, H, b, ...) except feat_H (M, b, D)
        return dict(
            actor_loss=loss,
            actor_grad_norm=gnorm,
            imag_return=jnp.mean(J),
            imag_reward=jnp.mean(traj["r"]),
            imag_term=jnp.mean(traj["d"]),
            imag_term_hard=jnp.mean(traj["d_hard"]),
            imag_saturation_penalty=jnp.mean(traj["sat"]),
            imag_survival=jnp.mean(traj["c"][:, -1]),
            **{f"imag_{n}": jnp.mean(traj["terms"][..., i]) for i, n in enumerate(core.TERM_NAMES)},
            **{f"imag_{n}": jnp.mean(traj["extra"][..., i]) for i, n in enumerate(core.EXTRA_NAMES)},
        )

    @partial(jax.jit, static_argnames=("horizon",))
    def ff_update(ff, ff_opt, actor_params, batch_stats, wm_params, critic_params, seeds, voc_now, horizon):
        """One gradient step on the feed-forward table through the world model; the policy stays frozen (PLAN.md 13.22.6)."""
        def loss_fn(f):
            def member(wm_p, sd, f_, cp):
                return rollout(actor_params, batch_stats, wm_p, cp, sd, horizon, voc_now, f_)

            J, traj = grouped(member, wm_params, seeds, f, critic_params)
            return -jnp.mean(J), (J, traj)

        (loss, (J, traj)), grads = jax.value_and_grad(loss_fn, has_aux=True)(ff)
        gnorm = optax.global_norm(grads)
        updates, ff_opt = actor_tx.update(grads, ff_opt, ff)
        return optax.apply_updates(ff, updates), ff_opt, jax.lax.stop_gradient(traj), imag_stats(loss, gnorm, J, traj)

    @jax.jit
    def critic_update(critic_params, critic_opt, traj, key):
        """Fit V to GAE(lambda) targets computed once over the imagined transitions (Eq. 3)."""
        # traj arrays are (M, H, b, ...) -> merge members into batch: (H, M*b, ...)
        def merge(x):
            return jnp.moveaxis(x, 0, 1).reshape((x.shape[1], -1) + x.shape[3:])

        feat = merge(traj["feat"])  # (H, B, D)
        feat_H = traj["feat_H"].reshape((-1,) + traj["feat_H"].shape[2:])
        r, d, trunc, c, alive = (merge(traj[n]) for n in ("r", "d", "trunc", "c", "alive"))
        H = feat.shape[0]
        values = critic_value(critic_params, jnp.concatenate([feat, feat_H[None]], 0))  # (H+1, B)

        def back(ret_next, t):
            cont = cfg.gamma * (1.0 - d[t]) * (1.0 - trunc[t])
            ret = r[t] + cont * ((1 - cfg.lam) * values[t + 1] + cfg.lam * ret_next)
            return ret, ret

        _, targets = jax.lax.scan(back, values[H], jnp.arange(H), reverse=True)
        weight = (alive * (c > 1e-3)).reshape(-1)
        X = feat.reshape(-1, feat.shape[-1])
        Y = targets.reshape(-1)
        N = X.shape[0]
        mb = N // cfg.critic_minibatches

        def loss_fn(p, idx):
            v = critic_value(p, X[idx])
            w = weight[idx]
            return jnp.sum(w * jnp.square(v - Y[idx])) / jnp.maximum(jnp.sum(w), 1.0)

        def epoch(carry, ekey):
            p, o = carry
            perm = jax.random.permutation(ekey, N)

            def minibatch(carry2, i):
                p2, o2 = carry2
                idx = jax.lax.dynamic_slice(perm, (i * mb,), (mb,))
                l, g = jax.value_and_grad(loss_fn)(p2, idx)
                u, o2 = critic_tx.update(g, o2, p2)
                return (optax.apply_updates(p2, u), o2), l

            (p, o), losses = jax.lax.scan(minibatch, (p, o), jnp.arange(cfg.critic_minibatches))
            return (p, o), jnp.mean(losses)

        (critic_params, critic_opt), losses = jax.lax.scan(
            epoch, (critic_params, critic_opt), jax.random.split(key, cfg.critic_epochs)
        )
        return critic_params, critic_opt, dict(critic_loss=losses[-1], value_mean=jnp.mean(values), target_mean=jnp.mean(Y))

    actor_tx = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.clip(cfg.grad_clip_value),
        optax.inject_hyperparams(optax.adam)(learning_rate=cfg.actor_lr, b1=0.9, b2=0.95),
    )
    critic_tx = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.inject_hyperparams(optax.adam)(learning_rate=cfg.critic_lr),
    )
    return dict(
        policy_mean=jax.jit(policy_mean),
        policy_ff=jax.jit(policy_ff),
        critic_value=jax.jit(critic_value),
        actor_update=actor_update,
        ff_update=ff_update,
        critic_update=critic_update,
        actor_tx=actor_tx,
        critic_tx=critic_tx,
    )


def set_lr(opt_state, lr):
    """Update the injected Adam learning rate (last element of the optax chain)."""
    inner = opt_state[-1]
    inner.hyperparams["learning_rate"] = jnp.asarray(lr, dtype=jnp.float32)
    return opt_state
