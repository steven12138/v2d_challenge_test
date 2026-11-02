"""SuperMimic training loop on one FlashCHORD ReconBody episode (Algorithm A.1 + object/finger extension)."""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json
import math
import pickle
import subprocess
import sys
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import warp as wp

from flash_chord.runtime.jax import from_jax, to_jax
from flash_chord.training.flash_sac.checkpoint import ActorInferenceState
from flash_chord.training.flash_sac.evaluation import compile_policy_action, create_actor_template
from flash_chord.training.flash_sac.network import Actor, project_unit_parameters

from supermimic import agent, buffer, core, env_io, export, residual, sonic_jax, world_model
from supermimic.config import Config

HERE = Path(__file__).resolve().parent


class Logger:
    def __init__(self, path: Path) -> None:
        self.file = path.open("a")

    def __call__(self, kind: str, **values) -> None:
        clean = {}
        for key, value in values.items():
            if isinstance(value, (jnp.ndarray, np.ndarray, np.generic)):
                value = np.asarray(value).tolist()
            clean[key] = value
        line = json.dumps({"kind": kind, "time": time.time(), **clean})
        self.file.write(line + "\n")
        self.file.flush()
        print(line, flush=True)


# ---------------------------------------------------------------------------------------------
# Collection


class Collector:
    """Steps the official env, logs pre/post-step rows into the replay buffer, and performs resets."""

    def __init__(self, cfg: Config, handles, k, readout, log):
        self.cfg = cfg
        self.env = handles.env
        self.k = k
        self.rd = readout
        self.log = log
        self.W = self.env.world_count
        self.C = cfg.buffer_steps
        self.buf = buffer.create(self.C, self.W)
        self.g = 0
        self.rng = np.random.default_rng(cfg.seed)
        self.key = jax.random.PRNGKey(cfg.seed + 1)
        self.reset_frame = wp.zeros(self.W, dtype=wp.int32)
        self.mask_explicit = wp.zeros(self.W, dtype=wp.int32)
        self.mask_sampled = wp.zeros(self.W, dtype=wp.int32)
        self.pack = jax.jit(partial(buffer.pack_raw, k))
        self.nbins = int(math.ceil((k["F"] - 1) / cfg.reset_bin_steps))
        self.bin_value = np.zeros(self.nbins)
        self.bin_hist = np.ones(self.nbins) / self.nbins
        self.g_max = 0.02 * 19.0 / (1.0 - cfg.gamma)
        # per-world tracking of frame-0 explicit episodes (closest proxy of the evaluation protocol)
        self.ep_frame0 = np.zeros(self.W, bool)
        self.ep_err = np.zeros(self.W)
        self.ep_len = np.zeros(self.W, int)
        self.window = dict(transitions=0, terminated=0, truncated=0, score=0.0, frame0_eps=[])
        self.obj_err = jax.jit(
            lambda sn, frame: jnp.linalg.norm(sn[:, 33:36] - k["ref_op"][jnp.clip(frame, 0, k["F"] - 1)], axis=-1)
        )
        self.weights = jnp.asarray(handles.env.objective.weights.numpy())
        lifted = k["ref_op"][:, 2] > k["ref_op"][0, 2] + 0.15  # reference object held off the ground
        self.lift0 = int(np.argmax(np.asarray(lifted))) if bool(jnp.any(lifted)) else k["F"]  # first lifted frame

        @jax.jit
        def grasp_stats(sn, frame, c_act):
            """Per world (4, W): link contact on reference-grasp frames (value, frame counted), and object kept
            near the reference while the reference holds it up (value, frame counted)."""
            f = jnp.clip(frame, 0, k["F"] - 1)
            grasping = jnp.sum(k["grasp_lab"][f], -1) > 0
            up = lifted[f]
            carried = up & (self.obj_err(sn, frame) < 0.15)
            return jnp.stack([core.grasp_touch(k, f, c_act) * grasping, grasping, carried, up]).astype(jnp.float32)

        self.grasp_stats = grasp_stats

        @jax.jit
        def contact_flags(frame, c_act):
            """Per world: the reference grasps now, and a grasping hand touches the object."""
            lab = k["grasp_lab"][jnp.clip(frame, 0, k["F"] - 1)]
            n = jnp.sum(c_act.reshape(-1, 2, 8), -1)
            return jnp.sum(lab, -1) > 0, jnp.sum(n * lab, -1) > 0

        self.contact_flags = contact_flags
        self.no_contact = np.zeros(self.W, int)  # consecutive grasp-frame steps without hand-object contact
        self.perf = dict(grasp=np.zeros(4), frame0_lens=[])  # VOC-decay gate statistics since the last check
        self.frame0_frac = cfg.frame0_reset_fraction
        self.diverged = jax.jit(lambda s, sn: ~(jnp.all(jnp.isfinite(s), -1) & jnp.all(jnp.isfinite(sn), -1)))
        self.limits = None  # policy_termination limits, set after the startup validation
        self.ep_off = np.full(self.W, -1)  # step at which a frame-0 episode first left the limits

        @jax.jit
        def over_limits(sn, frame):
            t = self.limits
            _, _, he, ho, oe, oo = core.tracking_errors(k, buffer.unpack_raw(sn), frame)
            palm = (jnp.max(he, -1) > t["palm_position"]) | (jnp.max(ho, -1) > t["palm_orientation"])
            return palm, (oe > t["object_position"]) | (oo > t["object_orientation"])

        self.over_limits = over_limits
        self.release_err = np.zeros(self.nbins)  # EMA of the palm error at the end of a settle, per frame bin
        self.bin_boost = np.ones(self.nbins)
        if self.lift0 < k["F"]:  # practise the approach and lift: windows that end just before the reference lift
            lo, hi = max(self.lift0 - 70, 0) // cfg.reset_bin_steps, max(self.lift0 - 15, 0) // cfg.reset_bin_steps
            self.bin_boost[lo : hi + 1] = cfg.pre_lift_boost
        held = np.nonzero(np.asarray(k["grasp_lab"]).sum(-1) > 0)[0]
        if len(held):  # practise the release: rollouts rarely get there, and a dropped object costs the end of the episode
            self.bin_boost[max(int(held[-1]) - 60, 0) // cfg.reset_bin_steps :] = cfg.release_boost
        self.palm_err = jax.jit(lambda sn, f: jnp.max(core.tracking_errors(k, buffer.unpack_raw(sn), f)[2], -1))
        self.voc_level = 1.0  # current VOC level, set by the training loop (gates the contact-loss termination)
        self.initial_reset()

    def initial_reset(self):
        done = np.ones(self.W, bool)
        self._reset(done)

    def step(self, policy, noise_std, extra=False):
        rd = self.rd
        env_io.sync()
        obs = jnp.nan_to_num(jnp.array(rd.observation()))
        self.key, sub = jax.random.split(self.key)
        action = jnp.nan_to_num(policy(obs))
        self.window["act_abs"] = self.window.get("act_abs", 0.0) + float(jnp.mean(jnp.abs(action)))
        self.window["act_sat"] = self.window.get("act_sat", 0.0) + float(jnp.mean(jnp.abs(action) > 9.0))
        self.window["steps"] = self.window.get("steps", 0) + 1
        action = jnp.clip(action + noise_std * jax.random.normal(sub, action.shape), -10.0, 10.0)
        if self.cfg.settle_policy_fingers_only:
            # The policy never trains on settling rows and pushes ahead of the frozen frame (30 cm off at release).
            # SONIC keeps the body at the reference; the fingers may still close on the object.
            settling = (jnp.array(rd.age()) < 51)[:, None]
            action = jnp.where(settling, action.at[:, :core.SONIC_DIM].set(0.0), action)
        pre = dict(
            s=self.pack(*rd.physics()),
            frame=jnp.array(rd.timestep()),
            estep=jnp.array(rd.episode_step()),
            age=jnp.array(rd.age()),
            start=jnp.array(rd.start_frame()),
            voc=jnp.array(rd.applied_voc()),
            lags=jnp.array(rd.lags()),
        )
        if extra:
            pre["obs"] = obs
            pre["tgt_hist"] = jnp.array(rd.target_history())
        flat = jnp.ravel(action)
        jax.block_until_ready((flat, pre))
        self.env.step(from_jax(flat))
        env_io.sync()
        post = dict(
            sn=self.pack(*rd.physics()),
            terms=jnp.array(rd.objective_terms()),
            term=jnp.array(rd.terminated()) != 0,
            trunc=jnp.array(rd.truncated()) != 0,
            c_act=jnp.array(rd.contact_active()).astype(jnp.float32),
            c_pos=jnp.array(rd.contact_pos()),
            c_dir=jnp.array(rd.contact_dir()),
            decoded=jnp.array(rd.sonic_decoded()),
            cur=jnp.array(rd.sonic_current()),
            target=jnp.array(rd.processed_target()),
            action=action,
        )
        if extra:
            post["dec_in"] = jnp.array(to_jax(self.env.action.decoder_input, (self.W, 994)))
        row = {**pre, **post}
        # One diverged world (contact solver blow-up) must not poison the replay buffer and with it every network: store
        # a healthy world's row in its place and end that episode.
        diverged = np.asarray(self.diverged(row["s"], row["sn"]))
        if diverged.any():
            self.window["diverged"] = self.window.get("diverged", 0) + int(diverged.sum())
            healthy = int(np.argmin(diverged))
            mask = jnp.asarray(diverged)
            row = {n: jnp.where(mask.reshape((-1,) + (1,) * (v.ndim - 1)), v[healthy], v) for n, v in row.items()}
            row["term"] = row["term"] | mask
        if self.cfg.contact_loss_steps > 0 and self.voc_level <= self.cfg.contact_loss_voc:
            # interaction early termination (InterMimic / ManipTrans): the reference holds the object but the
            # hands have not touched it for a while -> end the episode like any other tracking failure
            grasping, touching = (np.asarray(x) for x in self.contact_flags(row["frame"], row["c_act"]))
            self.no_contact = np.where(grasping & ~touching, self.no_contact + 1, 0)
            lost = (self.no_contact >= self.cfg.contact_loss_steps) & (np.asarray(row["estep"]) > self.k["term"]["freeze"])
            self.window["cause_contact"] = self.window.get("cause_contact", 0) + int(lost.sum())
            row["term"] = row["term"] | jnp.asarray(lost)
        else:
            self.no_contact[:] = 0
        if self.limits is not None:
            palm, obj = (np.asarray(x) for x in self.over_limits(row["sn"], row["frame"]))
            past = (palm | obj) & (np.asarray(row["estep"]) > self.k["term"]["freeze"])
            self.ep_off[past & self.ep_frame0 & (self.ep_off < 0)] = self.ep_len[past & self.ep_frame0 & (self.ep_off < 0)]
            lost = past & ~self.ep_frame0
            self.window["cause_palm"] = self.window.get("cause_palm", 0) + int((lost & palm).sum())
            self.window["cause_object"] = self.window.get("cause_object", 0) + int((lost & obj).sum())
            row["term"] = row["term"] | jnp.asarray(lost)
        jax.block_until_ready(row)
        self.buf = buffer.write(self.buf, self.g % self.C, self.g, {n: row[n] for n in buffer.FIELDS})
        self.g += 1

        term = np.asarray(row["term"])
        trunc = np.asarray(row["trunc"])
        causes = np.asarray(self.rd.term_causes()) != 0  # (W, 3) pelvis / palm / object
        cause_counts = causes[term].sum(axis=0)
        for i, name in enumerate(("pelvis", "palm", "object")):
            self.window[f"cause_{name}"] = self.window.get(f"cause_{name}", 0) + int(cause_counts[i])
        released = np.asarray(row["age"]) == 50
        if self.cfg.settle_reset_windows and released.any():
            errs = np.asarray(self.palm_err(row["sn"], row["frame"]))[released]
            for b, e in zip(np.asarray(row["frame"])[released] // self.cfg.reset_bin_steps, errs):
                b = min(int(b), self.nbins - 1)
                self.release_err[b] += 0.05 * (float(e) - self.release_err[b])
        err = np.asarray(self.obj_err(row["sn"], row["frame"]))
        gs = np.asarray(self.grasp_stats(row["sn"], row["frame"], row["c_act"]))  # (4, W)
        self.window["grasp"] = self.window.get("grasp", 0.0) + gs.sum(-1)
        self.perf["grasp"] += gs[:, self.ep_frame0].sum(-1)  # the gate looks at evaluation-like episodes only
        self.ep_err += err
        self.ep_len += 1
        self.window["transitions"] += self.W
        self.window["terminated"] += int(term.sum())
        self.window["truncated"] += int(trunc.sum())
        self.window["score"] += float(jnp.sum(0.02 * row["terms"] @ self.weights))
        done = term | trunc
        for w in np.nonzero(done & self.ep_frame0)[0]:
            self.window["frame0_eps"].append(
                (int(self.ep_len[w]), float(self.ep_err[w] / max(self.ep_len[w], 1)), bool(trunc[w]))
            )
            self.perf["frame0_lens"].append(int(self.ep_len[w]))
            self.window.setdefault("frame0_off", []).append(int(self.ep_off[w] if self.ep_off[w] >= 0 else self.ep_len[w]))
        self._reset(done)
        return row

    def _reset(self, done):
        if not done.any():
            return
        cfg = self.cfg
        u = self.rng.random(self.W)
        frame0 = done & (u < self.frame0_frac)
        returned_frac = 0.0 if cfg.settle_reset_windows else cfg.return_reset_fraction
        explicit = done & (u < self.frame0_frac + returned_frac)
        sampled = done & ~explicit
        returned = explicit & ~frame0
        frames = np.zeros(self.W, np.int32)
        n_ret = int(returned.sum())
        if n_ret:
            soft = np.exp(-25.0 * self.bin_value / self.g_max)
            soft = soft / soft.sum()
            mix = 0.5 / self.nbins + 0.4 * soft + 0.1 * self.bin_hist
            mix = mix / mix.sum()
            bins = self.rng.choice(self.nbins, size=n_ret, p=mix)
            offs = self.rng.integers(0, cfg.reset_bin_steps, size=n_ret)
            frames[returned] = np.minimum(bins * cfg.reset_bin_steps + offs, self.k["F"] - 2)
        if explicit.any():
            self.reset_frame.assign(frames)
            self.mask_explicit.assign(explicit.astype(np.int32))
            self.env.reset(reset_frame=self.reset_frame, reset_mask=self.mask_explicit)
        if sampled.any():
            if cfg.settle_reset_windows:
                self._choose_window(uniform=sampled.sum() > self.W // 4)
            self.mask_sampled.assign(sampled.astype(np.int32))
            self.env.reset(reset_mask=self.mask_sampled)
        self.ep_frame0[done] = frame0[done]
        self.ep_err[done] = 0.0
        self.ep_len[done] = 0
        self.no_contact[done] = 0
        self.ep_off[done] = -1

    def _choose_window(self, uniform):
        """The settle reset samples one frame window per call: draw it from the return-based mix of Algorithm A.1,
        favouring the lift and avoiding frames whose frozen pose falls apart during the settle."""
        policy = self.env.reset_policy
        if uniform:
            policy.start_frame, policy.end_frame = 0, self.k["F"] - 1
            return
        soft = np.exp(-25.0 * self.bin_value / self.g_max)
        mix = 0.5 / self.nbins + 0.4 * soft / soft.sum() + 0.1 * self.bin_hist
        mix = mix * self.bin_boost * (0.1 + 0.9 * (self.release_err < self.cfg.settle_ok_palm))
        b = int(self.rng.choice(self.nbins, p=mix / mix.sum()))
        policy.start_frame = b * self.cfg.reset_bin_steps
        policy.end_frame = min(policy.start_frame + self.cfg.reset_bin_steps - 1, self.k["F"] - 1)

    def update_bins(self, values_by_bin):
        self.bin_value = values_by_bin
        soft = np.exp(-25.0 * self.bin_value / self.g_max)
        self.bin_hist = 0.95 * self.bin_hist + 0.05 * soft / soft.sum()

    def pop_window(self):
        w = self.window
        eps = w["frame0_eps"]
        out = dict(
            transitions=w["transitions"],
            term_rate=w["terminated"] / max(w["transitions"], 1),
            trunc_rate=w["truncated"] / max(w["transitions"], 1),
            score_per_step=w["score"] / max(w["transitions"], 1),
            frame0_episodes=len(eps),
            frame0_len=float(np.mean([e[0] for e in eps])) if eps else None,
            frame0_obj_err=float(np.mean([e[1] for e in eps])) if eps else None,
            frame0_full=float(np.mean([e[2] for e in eps])) if eps else None,
            term_cause_pelvis=w.get("cause_pelvis", 0) / max(w["terminated"], 1),
            term_cause_palm=w.get("cause_palm", 0) / max(w["terminated"], 1),
            term_cause_object=w.get("cause_object", 0) / max(w["terminated"], 1),
            term_cause_contact=w.get("cause_contact", 0) / max(w["terminated"], 1),
            policy_abs_action=w.get("act_abs", 0.0) / max(w.get("steps", 1), 1),
            policy_saturated=w.get("act_sat", 0.0) / max(w.get("steps", 1), 1),
        )
        off = w.get("frame0_off", [])
        out.update(frame0_off=float(np.mean(off)) if off else None, diverged_worlds=w.get("diverged", 0))
        g = w.get("grasp", np.zeros(4))
        out.update(sim_grasp_touch=float(g[0] / max(g[1], 1)), sim_carry_ok=float(g[2] / max(g[3], 1)),
                   sim_lifted_rows=float(g[3] / max(w["transitions"], 1)),
                   release_palm_err=float(np.median(self.release_err)),
                   release_ok_bins=float(np.mean(self.release_err < self.cfg.settle_ok_palm)))
        self.window = dict(transitions=0, terminated=0, truncated=0, score=0.0, frame0_eps=[])
        return out

    def pop_perf(self):
        p = self.perf
        g = p["grasp"]
        out = dict(frame0_episodes=len(p["frame0_lens"]),
                   len_mean=float(np.mean(p["frame0_lens"])) if p["frame0_lens"] else 0.0,
                   len_frac=float(np.mean(p["frame0_lens"]) / (self.k["F"] - 1)) if p["frame0_lens"] else 0.0,
                   carry=float(g[2] / max(g[3], 1)), touch=float(g[0] / max(g[1], 1)))
        self.perf = dict(grasp=np.zeros(4), frame0_lens=[])
        return out

    @property
    def g_lo(self):
        return max(0, self.g - self.C)

    @property
    def g_hi(self):
        return self.g - 1


# ---------------------------------------------------------------------------------------------
# Startup validation of every JAX re-implementation against the live environment


def validate_against_env(k, dec, rows, log, n_worlds=256):
    rows = [{n: v[:n_worlds] for n, v in r.items()} for r in rows]
    T = len(rows)
    stack = {n: jnp.stack([r[n] for r in rows]) for n in rows[0]}
    hp = jax.lax.Precision.HIGHEST
    report = {}

    # 1. SONIC decoder port vs ONNX Runtime on the exact logged decoder input
    pred = jax.vmap(lambda x: core.sonic_decode(dec, x, precision=hp))(stack["dec_in"])
    report["sonic_decoder_max_err"] = float(jnp.max(jnp.abs(pred - stack["decoded"])))
    # ONNX Runtime's CUDA EP runs TF32 matmuls: compare relative to the output magnitude
    rel = jnp.abs(pred - stack["decoded"]) / (1.0 + jnp.abs(stack["decoded"]))
    report["sonic_decoder_max_rel_err"] = float(jnp.max(rel))
    report["sonic_decoder_mean_rel_err"] = float(jnp.mean(rel))

    settled = np.asarray(stack["age"] >= 40)  # reference offsets are zero from age 40 on
    estep_all = np.asarray(stack["estep"])
    worlds = jnp.arange(stack["s"].shape[1])
    obs_err = np.zeros(core.OBS_DIM)
    cur_err, hist_err, latent_mis, tgt_err = 0.0, 0.0, [], 0.0
    term_err = np.zeros(12)
    fc_errs, term_agree = [], []
    for t in range(T):
        r = rows[t]
        m = settled[t]
        if not m.any():
            continue
        S = buffer.unpack_raw(r["s"])
        Sn = buffer.unpack_raw(r["sn"])
        estep = estep_all[t]
        has_prev = jnp.asarray(estep > 0)[:, None]
        obs = core.observation(k, S, r["frame"], r["tgt_hist"])
        obs_err = np.maximum(obs_err, np.asarray(jnp.max(jnp.abs(obs - r["obs"])[m], axis=0)))
        prev_dec = rows[t - 1]["decoded"] if t > 0 else k["sonic_default"][None]
        last_dec = jnp.where(has_prev, prev_dec, k["sonic_default"][None])
        cur = core.sonic_current(k, S, last_dec)
        cur_err = max(cur_err, float(jnp.max(jnp.abs(cur - r["cur"])[m])))
        hist = [stack["cur"][jnp.asarray(np.maximum(t - kk, t - estep)), worlds] for kk in range(9, 0, -1)]
        hist = jnp.stack(hist + [r["cur"]], axis=1)
        din = core.sonic_decoder_input(k["latent"][jnp.clip(r["frame"], 0, k["F"] - 1)], hist)
        diff = jnp.abs(din - r["dec_in"])[m]
        hist_err = max(hist_err, float(jnp.max(diff[:, 64:])))
        latent_mis.append(float(jnp.mean((diff[:, :64] > 1e-4).astype(jnp.float32))))
        tgt = core.processed_targets(k, r["decoded"], r["action"], r["frame"])
        tgt_err = max(tgt_err, float(jnp.max(jnp.abs(tgt - r["target"])[m])))
        prev_a = jnp.where(has_prev, rows[t - 1]["action"] if t > 0 else 0.0, 0.0)
        terms = core.objective_terms(k, Sn, r["frame"], r["start"], r["action"], prev_a, r["terms"][:, 11],
                                     r["term"].astype(jnp.float32))
        rel = jnp.abs(terms - r["terms"]) / (1.0 + jnp.abs(r["terms"]))
        term_err = np.maximum(term_err, np.asarray(jnp.max(rel[m], axis=0)))
        fc = core.force_closure(k, r["frame"], jnp.ones_like(r["c_act"]), r["c_pos"], r["c_dir"], sharp=None)
        fc_errs.append(np.asarray(jnp.abs(fc - r["terms"][:, 11])[m]))
        hard = core.termination_hard(k, Sn, r["frame"], r["estep"])
        term_agree.append(np.asarray(hard == r["term"])[m])

    from flash_chord.envs.recon_body import RECON_BODY_OBSERVATION_BLOCKS

    blocks, cursor = [], 0
    for name, width in RECON_BODY_OBSERVATION_BLOCKS:
        blocks.append((name, float(obs_err[cursor : cursor + width].max())))
        cursor += width
    fc_all = np.concatenate(fc_errs)
    report.update(
        obs_block_max_err=blocks,
        obs_max_err=float(obs_err.max()),
        sonic_current_max_err=cur_err,
        decoder_input_history_max_err=hist_err,
        latent_mismatch_frac=float(np.mean(latent_mis)),
        target_max_err=tgt_err,
        objective_term_max_rel_err=dict(zip(core.TERM_NAMES, term_err.tolist())),
        force_closure_mean_err=float(fc_all.mean()),
        force_closure_max_err=float(fc_all.max()),
        termination_agreement=float(np.concatenate(term_agree).mean()),
        settled_rows=int(settled.sum()),
    )
    log("validation", **report)

    failures = []
    # the max over ~15k decoder calls is a TF32 tail (0.003-0.016 observed); a real port bug shows in the mean
    if report["sonic_decoder_mean_rel_err"] > 2e-3 or report["sonic_decoder_max_rel_err"] > 5e-2:
        failures.append("sonic decoder")
    if report["obs_max_err"] > 5e-3:
        failures.append("observation")
    if cur_err > 1e-3:
        failures.append("sonic current observation")
    if hist_err > 1e-3:
        failures.append("sonic history reconstruction")
    if report["latent_mismatch_frac"] > 0.02:
        failures.append("sonic latent table")
    if tgt_err > 1e-3:
        failures.append("processed targets")
    bad_terms = [n for n, e in report["objective_term_max_rel_err"].items()
                 if n not in ("termination", "force_closure") and e > 2e-3]
    if bad_terms:
        failures.append(f"objective terms {bad_terms}")
    if report["force_closure_mean_err"] > 5e-3:
        failures.append("force closure")
    if report["termination_agreement"] < 0.995:
        failures.append("termination")
    if failures:
        raise AssertionError(f"re-implementation does not match the environment: {failures}")


# ---------------------------------------------------------------------------------------------
# Sampling from the replay buffer


def make_samplers(k, cfg, W, C):
    K = cfg.wm_window
    F = k["F"]

    def windows(buf, g_lo, g_hi, key, length, batch):
        k1, k2 = jax.random.split(key)
        g0 = jax.random.randint(k1, (batch,), g_lo + 1, jnp.maximum(g_hi - length + 2, g_lo + 2))
        w = jax.random.randint(k2, (batch,), 0, W)
        gs = g0[None] + jnp.arange(length)[:, None]
        slot = jnp.mod(gs, C)
        rows = {n: buf[n][slot, w[None]] for n in buffer.FIELDS}
        rows["prev_target"] = buf["target"][jnp.mod(gs - 1, C), w[None]]
        present = buf["g"][slot] == gs
        same_ep = rows["estep"] == rows["estep"][0:1] + jnp.arange(length)[:, None]
        done = (rows["term"] | rows["trunc"]).astype(jnp.int32)
        done_before = jnp.cumsum(done, axis=0) - done
        rows["valid"] = (present & same_ep & (done_before == 0)).astype(jnp.float32)
        return rows

    def seeds(buf, g_lo, g_hi, key, n, frame0_frac):
        k1, k2, k3 = jax.random.split(key, 3)
        cand = 8 * n
        g = jax.random.randint(k1, (cand,), g_lo + 10, g_hi + 1)
        w = jax.random.randint(k2, (cand,), 0, W)
        slot = jnp.mod(g, C)
        age, frame, start = buf["age"][slot, w], buf["frame"][slot, w], buf["start"][slot, w]
        valid = (age >= 51) & (frame < F - 1) & (buf["g"][slot] == g)
        u = jax.random.uniform(k3, (cand,))
        n0 = int(n * frame0_frac)
        _, i0 = jax.lax.top_k((valid & (start == 0)).astype(jnp.float32) + 0.5 * u, n0)
        _, i1 = jax.lax.top_k(valid.astype(jnp.float32) + 0.5 * u, n)
        idx = jnp.concatenate([i0, i1[: n - n0]])
        g, w = g[idx], w[idx]
        slot = jnp.mod(g, C)
        estep = buf["estep"][slot, w]
        hist = []
        for kk in range(9, 0, -1):
            gi = jnp.maximum(g - kk, g - estep)
            hist.append(buf["cur"][jnp.mod(gi, C), w])
        tgt_hist = []
        for kk in range(3, 0, -1):
            tgt_hist.append(jnp.where((kk <= estep)[:, None], buf["target"][jnp.mod(g - kk, C), w], 0.0))
        has_prev = (estep > 0)[:, None]
        prev = jnp.mod(g - 1, C)
        return dict(
            s=buf["s"][slot, w],
            lags=buf["lags"][slot, w],
            frame=buf["frame"][slot, w],
            estep=estep,
            start=buf["start"][slot, w],
            voc=buf["voc"][slot, w],
            hist9=jnp.stack(hist, axis=1),
            last_dec=jnp.where(has_prev, buf["decoded"][prev, w], k["sonic_default"][None]),
            tgt_hist=jnp.stack(tgt_hist, axis=1),
            prev_a=jnp.where(has_prev, buf["action"][prev, w], 0.0),
            prev_tgt=buf["target"][prev, w],
            first=estep == 0,
            valid=valid[idx],
        )

    return windows, seeds


# ---------------------------------------------------------------------------------------------


def schedule_voc(cfg, frac):
    """Piecewise-linear VOC curriculum (small steps keep the world model in distribution)."""
    xs = [x for x, _ in cfg.voc_schedule]
    ys = [y for _, y in cfg.voc_schedule]
    return float(np.interp(frac, xs, ys))


def piecewise(schedule, frac, default):
    """Value of a piecewise-constant [[frac, value], ...] schedule."""
    value = default
    for start, v in schedule:
        if frac >= start:
            value = float(v)
    return value


def cosine(frac, final_ratio):
    return final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0)))


def set_batch_stats(batch_stats, name, mean, var):
    mutable = flax.core.unfreeze(batch_stats) if isinstance(batch_stats, flax.core.FrozenDict) else jax.tree.map(lambda x: x, batch_stats)
    mutable[name]["running_mean"] = jnp.asarray(mean, jnp.float32)
    mutable[name]["running_variance"] = jnp.asarray(var, jnp.float32)
    return flax.core.freeze(mutable) if isinstance(batch_stats, flax.core.FrozenDict) else mutable


def launch_eval(ckpt: Path, out: Path, gpu: str, log, episode: int = 8):
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["bash", str(HERE / "eval_episode.sh"), str(ckpt), str(out), str(episode)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    proc = subprocess.Popen(cmd, stdout=(out / "eval.log").open("w"), stderr=subprocess.STDOUT, env=env)
    log("eval_launched", checkpoint=str(ckpt), out=str(out), pid=proc.pid)
    return proc


def parse_overrides(cfg: Config, argv) -> Config:
    """`python -m supermimic.main key=value ...` overrides any Config field (typed by its default)."""
    for arg in argv:
        name, value = arg.split("=", 1)
        current = getattr(cfg, name)
        if isinstance(current, bool):
            value = value.lower() in ("1", "true", "yes")
        elif isinstance(current, (int, float, str)):
            value = type(current)(value)
        else:
            value = json.loads(value)
        setattr(cfg, name, value)
    return cfg


def main():
    cfg = parse_overrides(Config(), sys.argv[1:])
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / "train.jsonl")
    log("config", **{k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()})
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    with wp.ScopedDevice("cuda:0"):
        t_build = time.time()
        handles = env_io.build(cfg.parquet, cfg.world_count, voc_gravity_comp=cfg.voc_gravity_comp,
                               voc_angular_scale=cfg.voc_angular_scale,
                               termination_overrides=cfg.termination_overrides, voc_k_lin=cfg.voc_k_lin,
                               voc_d_lin=cfg.voc_d_lin)
        env = handles.env
        consts = env_io.extract_consts(handles)
        consts.latent_table = sonic_jax.latent_table(env)
        k = core.device_consts(consts)
        k["delta_w"] = core.delta_weights(cfg.wm_object_weight)
        k["grip_q"] = jnp.asarray(core.finger_grip_table(consts, cfg.finger_grip_frac, release_lead=cfg.grip_release_lead), jnp.float32)
        k["leg_dims"], k["arm_dims"] = core.residual_groups(consts)
        dec = sonic_jax.load_decoder_params()
        rd = env_io.Readout(env, consts)
        W, F = env.world_count, k["F"]
        if env.contact.layout.slots_per_world != 16 or env.action.action_dim != 43 or env.observation_dim != 486:
            raise RuntimeError("unexpected ReconBody layout")
        log("built", seconds=time.time() - t_build, frames=F, worlds=W, bodies=consts.bodies_per_world,
            voc_runtime=consts.voc, termination_runtime=consts.termination,
            q_per_world=consts.q_per_world, contact_sides=list(consts.contact_sides), obj_mass=consts.obj_mass)

        training = handles.training
        base_training, base_hidden = training, training.network.actor_hidden_dim
        checkpoint_meta = handles.checkpoint_metadata
        if cfg.residual_hidden > 0:
            if not cfg.resume_policy:
                raise ValueError("residual_hidden needs the base policy in resume_policy")
            training = residual.widen_training(training, cfg.residual_hidden)
            checkpoint_meta = residual.widen_metadata(checkpoint_meta, base_hidden + cfg.residual_hidden)
        actor32 = Actor(action_dim=43, config=training.network, dtype=jnp.float32)
        actor16, template = create_actor_template(training, 486, 43)
        actor_params, batch_stats = template.params, template.batch_stats
        resume = None
        if cfg.resume_state:
            with open(cfg.resume_state, "rb") as fh:
                resume = jax.tree.map(jnp.asarray, pickle.load(fh))
            actor_params, batch_stats = resume["actor_params"], resume["batch_stats"]
            log("resume", path=cfg.resume_state, frac=float(resume["frac"]))
        elif cfg.resume_policy:
            from flash_chord.training.flash_sac.checkpoint import load_actor_checkpoint

            if cfg.residual_hidden > 0:
                base, _ = load_actor_checkpoint(cfg.resume_policy, create_actor_template(base_training, 486, 43)[1])
                actor_params = residual.widen_params(base.params, template.params, base_hidden)
                batch_stats = residual.widen_stats(base.batch_stats, template.batch_stats, base_hidden)
            else:
                loaded, _ = load_actor_checkpoint(cfg.resume_policy, template)
                actor_params, batch_stats = loaded.params, loaded.batch_stats
            log("resume_policy", path=cfg.resume_policy, start_frac=cfg.start_frac, residual_hidden=cfg.residual_hidden)
        key = jax.random.PRNGKey(cfg.seed)
        dummy_obs = jnp.zeros((W, 486))
        policy16 = compile_policy_action(actor16, template, dummy_obs, key, deterministic=True,
                                         action_transform="tanh", action_scale=10.0)

        def collection_policy(params, stats):
            state = ActorInferenceState(params=params, batch_stats=stats)
            return lambda obs: policy16(state, obs, key)[0]

        # A resumed world model must warm up on the saved assistance level.
        if cfg.resume_voc_level > 1.0:
            raise ValueError("resume_voc_level must be in [0, 1]")
        initial_voc = (cfg.resume_voc_level if cfg.resume_voc_level >= 0.0 else
                       float(resume.get("voc_level", 1.0)) if resume is not None else 1.0)
        env.set_voc_scale(initial_voc)
        collector = Collector(cfg, handles, k, rd, log)

        # ---- 1. validation rollouts (zero residual + noise) and re-implementation checks
        t0 = time.time()
        vrows = [collector.step(collection_policy(actor_params, batch_stats), cfg.action_noise, extra=True) for _ in range(60)]
        log("collect_speed", steps_per_second=60 / (time.time() - t0))
        validate_against_env(k, dec, vrows, log)
        del vrows
        if cfg.policy_termination:
            k["term"].update(cfg.policy_termination)  # imagination and collector limits; the env only ends falls
            collector.limits = dict(k["term"])

        # ---- 2. warmup
        obs_samples = []
        t0 = time.time()
        for i in range(cfg.warmup_steps):
            collector.step(collection_policy(actor_params, batch_stats), cfg.action_noise)
            if i % 4 == 0:
                obs_samples.append(jnp.array(rd.observation())[::4])
        log("warmup_done", seconds=time.time() - t0, rows=collector.g)
        obs_all = jnp.concatenate(obs_samples)
        if resume is None and not cfg.resume_policy:  # a resumed policy keeps its own input normalization
            batch_stats = set_batch_stats(batch_stats, "input_norm", jnp.mean(obs_all, 0),
                                          jnp.maximum(jnp.var(obs_all, 0), cfg.obs_var_floor))
        if cfg.residual_hidden > 0:
            batch_stats = residual.calibrate(actor32, actor_params, batch_stats, obs_all[:: max(obs_all.shape[0] // 8192, 1)], base_hidden)
        del obs_samples, obs_all

        sample_windows, sample_seeds = make_samplers(k, cfg, W, cfg.buffer_steps)
        nkey = jax.random.PRNGKey(cfg.seed + 7)
        rows1 = jax.jit(partial(sample_windows, length=1, batch=65536))(collector.buf, collector.g_lo, collector.g_hi, nkey)
        r0 = {n: v[0] for n, v in rows1.items()}
        m = r0["valid"] > 0
        S = buffer.unpack_raw(r0["s"])
        Sn = buffer.unpack_raw(r0["sn"])
        prev = jnp.where((r0["estep"] > 0)[:, None], r0["prev_target"], r0["target"])
        x_in = core.wm_inputs(k, S, r0["target"], prev, core.lag_onehot(r0["lags"]), r0["voc"], r0["frame"])[m]
        y_out = core.encode_delta(S, Sn)[m]
        wm_norm = {"in": world_model.normalizer_from(x_in), "out": world_model.normalizer_from(y_out, floor=1e-4)}
        cfeat = core.critic_features(k, S, r0["frame"], r0["estep"], r0["voc"], core.lag_onehot(r0["lags"]))[m]
        critic_norm = world_model.normalizer_from(cfeat)
        if resume is not None:
            wm_norm, critic_norm = resume["wm_norm"], resume["critic_norm"]
        del rows1, r0, S, Sn, x_in, y_out, cfeat

        # ---- 3. world model + agent functions
        wm_params = world_model.init(jax.random.PRNGKey(cfg.seed + 11), cfg.wm_members, cfg.wm_hidden, cfg.wm_layers)
        wm_tx = optax.chain(optax.clip_by_global_norm(1.0), optax.inject_hyperparams(optax.adam)(learning_rate=cfg.wm_lr))
        wm_opt = wm_tx.init(wm_params)

        def wm_member_loss(p, rows):
            return world_model.window_loss(p, k, wm_norm, rows)

        @jax.jit
        def wm_update(params, opt, buf, g_lo, g_hi, key):
            keys = jax.random.split(key, cfg.wm_members)
            rows = jax.vmap(lambda kk: sample_windows(buf, g_lo, g_hi, kk, cfg.wm_window, cfg.wm_batch))(keys)

            def loss_fn(p):
                losses, parts = jax.vmap(wm_member_loss)(p, rows)
                return jnp.sum(losses), parts

            (loss, parts), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt = wm_tx.update(grads, opt, params)
            return optax.apply_updates(params, updates), opt, dict(wm_loss=loss / cfg.wm_members,
                                                                  **{f"wm_{n}": jnp.mean(v) for n, v in parts.items()})

        @jax.jit
        def wm_openloop(params, buf, g_lo, g_hi, key):
            """Open-loop 32-step error of member 0 on logged actions (object / palm position)."""
            rows = sample_windows(buf, g_lo, g_hi, key, 32, 1024)
            p0 = jax.tree.map(lambda x: x[0], params)
            S = buffer.unpack_raw(rows["s"][0])
            obj, palm, held = [], [], []
            in_contact = jnp.sum(rows["c_act"][0], -1) > 0  # what the grasp gradient depends on
            for j in range(32):
                prev = jnp.where((rows["estep"][j] > 0)[:, None], rows["prev_target"][j], rows["target"][j])
                S, _, _, _ = world_model.step(p0, k, wm_norm, S, rows["target"][j], prev,
                                              core.lag_onehot(rows["lags"][j]), rows["voc"][j], rows["frame"][j])
                St = buffer.unpack_raw(rows["sn"][j])
                vm = rows["valid"][j]
                obj.append(jnp.sum(jnp.linalg.norm(S["op"] - St["op"], axis=-1) * vm) / jnp.maximum(jnp.sum(vm), 1))
                held.append(jnp.sum(jnp.linalg.norm(S["op"] - St["op"], axis=-1) * vm * in_contact)
                            / jnp.maximum(jnp.sum(vm * in_contact), 1))
                pp, _ = core.palm_frames(k, S)
                pt, _ = core.palm_frames(k, St)
                palm.append(jnp.sum(jnp.mean(jnp.linalg.norm(pp - pt, axis=-1), -1) * vm) / jnp.maximum(jnp.sum(vm), 1))
            return dict(wm_obj_err_8=obj[7], wm_obj_err_32=obj[31], wm_palm_err_8=palm[7], wm_palm_err_32=palm[31],
                        wm_held_obj_err_8=held[7], wm_held_obj_err_32=held[31])

        weights = jnp.asarray(env.objective.weights.numpy())
        if cfg.w_action_rate != 0.0:
            weights = weights.at[core.TERM_NAMES.index("action_rate_l2")].set(cfg.w_action_rate)
        extra_weights = jnp.asarray([cfg.w_object_orientation, cfg.w_object_position_fine, cfg.w_hand_in_object,
                                     cfg.w_grasp_contact, cfg.w_object_velocity, cfg.w_object_position_coarse,
                                     cfg.w_pelvis_angular_stability, cfg.w_torso_sideways_lean,
                                     cfg.w_pelvis_velocity, cfg.w_pelvis_angular_velocity, cfg.w_body_joint_velocity,
                                     cfg.w_finger_grip, cfg.w_action_residual, cfg.w_object_orientation_coarse,
                                     cfg.w_residual_legs, cfg.w_residual_arms, cfg.w_object_keypoints, cfg.w_object_keypoints_fine])
        trainable = residual.trainable_mask(actor_params, base_hidden) if cfg.residual_hidden > 0 else None
        fns = agent.make_fns(k, cfg, actor32, dec, wm_norm, critic_norm, weights, extra_weights, trainable)
        actor_opt = fns["actor_tx"].init(actor_params)
        critic_params = agent.critic_init(jax.random.PRNGKey(cfg.seed + 13), cfg.critic_hidden)
        critic_opt = fns["critic_tx"].init(critic_params)
        seed_fns = {}

        def seeds_for(frac0):
            """The frame-0 share is a static top_k size: one compiled sampler per schedule value."""
            if frac0 not in seed_fns:
                seed_fns[frac0] = jax.jit(partial(sample_seeds, n=cfg.imag_batch, frame0_frac=frac0))
            return seed_fns[frac0]

        seeds_fn = seeds_for(cfg.frame0_seed_fraction)
        start_frac = cfg.start_frac
        if resume is not None:
            wm_params, wm_opt = resume["wm_params"], resume["wm_opt"]
            critic_params, critic_opt = resume["critic_params"], resume["critic_opt"]
            actor_opt = resume["actor_opt"]
            collector.bin_value = np.asarray(resume["bin_value"])
            collector.bin_hist = np.asarray(resume["bin_hist"])
            start_frac = float(resume["frac"])
            if cfg.resume_schedule_frac >= 0.0:
                if cfg.resume_schedule_frac >= min(1.0, cfg.stop_frac):
                    raise ValueError("resume_schedule_frac must be below stop_frac")
                start_frac = cfg.resume_schedule_frac
                log("resume_schedule", saved_frac=float(resume["frac"]), new_frac=start_frac)
        voc_level = cfg.voc_start
        if resume is not None:
            voc_level = float(resume.get("voc_level", schedule_voc(cfg, start_frac)))
            if cfg.resume_voc_level >= 0.0:
                voc_level = cfg.resume_voc_level
        gate_len = cfg.voc_gate_len
        if cfg.voc_gate_after_lift > 0:
            gate_len = min(1.0, (collector.lift0 + cfg.voc_gate_after_lift) / (F - 1))
        gate_fails = 0  # consecutive checks in which frame-0 rollouts die before the lift is underway

        # ---- 4. world-model pretraining on warmup data
        t0 = time.time()
        wkey = jax.random.PRNGKey(cfg.seed + 17)
        for i in range(cfg.wm_pretrain_updates if resume is None else 300):
            wkey, sub = jax.random.split(wkey)
            wm_params, wm_opt, wm_stats = wm_update(wm_params, wm_opt, collector.buf, collector.g_lo, collector.g_hi, sub)
            if i % 500 == 0 or i == cfg.wm_pretrain_updates - 1:
                wkey, sub = jax.random.split(wkey)
                ol = wm_openloop(wm_params, collector.buf, collector.g_lo, collector.g_hi, sub)
                log("wm_pretrain", update=i, seconds=time.time() - t0, **jax.device_get({**wm_stats, **ol}))

        # ---- 5. Dyna loop
        meta = checkpoint_meta
        exports, evals = [], []
        # SONIC-only reference checkpoint (zero residual) for the comparison
        if not cfg.final_only:
            ref = export.export(out / "checkpoints" / "policy_sonic_only.safetensors", template.params, template.batch_stats,
                                meta, training, 0, jnp.zeros((64, 486)), fns["policy_mean"])
            log("export", **ref)
            if resume is None and not cfg.resume_policy:
                evals.append(launch_eval(Path(ref["path"]), out / "eval" / "sonic_only", gpu, log, cfg.episode_index))

        total = cfg.train_hours * 3600.0
        t_start = time.time()
        last_export = t_start
        n_loop = 0  # iterations run by this process (iteration-based schedule; a shared GPU only slows it down)
        it = 0 if resume is None else cfg.critic_warmup_iters  # a resumed critic needs no warmup
        akey = jax.random.PRNGKey(cfg.seed + 19)
        timing = dict(collect=0.0, world_model=0.0, actor=0.0, critic=0.0)
        ema_params = actor_params
        anchor_params = actor_params
        ff, ff_opt, ema_ff = None, None, None
        if cfg.ff_knot_frames > 0:
            ff = jnp.zeros(((F - 1) // cfg.ff_knot_frames + 2, 43), jnp.float32)
            ff_opt, ema_ff = fns["actor_tx"].init(ff), ff

        def save_ff(name):
            np.savez(out / "checkpoints" / f"ff_{name}.npz", ff=np.asarray(ff), ema_ff=np.asarray(ema_ff), knot_frames=cfg.ff_knot_frames)

        ema_update = jax.jit(lambda e, p: jax.tree.map(lambda a, b: cfg.policy_ema_decay * a + (1 - cfg.policy_ema_decay) * b, e, p))

        def export_ema(name):
            """The policy swings between checkpoints at VOC 0 (ep 3: 48 % -> 25 % of worlds within 10 cm in 1000 iterations)."""
            res = export.export(out / "checkpoints" / f"policy_ema_{name}.safetensors",
                                ema_params if trainable is not None else project_unit_parameters(ema_params), batch_stats,
                                meta, training, collector.g * W, jnp.array(rd.observation())[:256], fns["policy_mean"])
            log("export", ema=True, **res)

        while True:
            elapsed = time.time() - t_start
            frac = start_frac + (n_loop / cfg.train_iters if cfg.train_iters > 0 else elapsed / total)
            if frac >= min(1.0, cfg.stop_frac):
                break
            voc_now = schedule_voc(cfg, frac)
            if cfg.voc_adaptive:  # the schedule is only an upper bound; checks below lower the level earlier
                voc_level = min(voc_level, voc_now)
                voc_now = voc_level
                if cfg.voc_stage_stop > 0 and voc_level <= cfg.voc_stage_stop:
                    log("voc_stage_stop", level=voc_level, frac=frac)
                    break
            collector.voc_level = voc_now
            collector.frame0_frac = piecewise(cfg.frame0_reset_schedule, frac, cfg.frame0_reset_fraction)
            seeds_fn = seeds_for(piecewise(cfg.frame0_seed_schedule, frac, cfg.frame0_seed_fraction))
            if abs(float(env.voc_scale.numpy()[0]) - voc_now) > 2e-3:
                previous = float(env.voc_scale.numpy()[0])
                env.set_voc_scale(voc_now)
                if int(previous * 20) != int(voc_now * 20):
                    log("voc", value=voc_now, frac=frac)
            horizon = cfg.horizon_short if frac < cfg.horizon_switch_frac else cfg.horizon_long
            lr_scale = cosine(frac, cfg.lr_final_ratio)
            actor_lr, ff_lr = cfg.actor_lr * lr_scale, cfg.ff_lr * lr_scale
            if cfg.actor_lr_peak > 0:  # warmup from 0 once the critic is ready, then cosine down over the rest of this run
                trained = max(n_loop - cfg.critic_warmup_iters, 0)
                total_iters = max(cfg.train_iters * (min(1.0, cfg.stop_frac) - start_frac) - cfg.critic_warmup_iters, 1)
                past = min(max(trained - cfg.actor_lr_warmup, 0) / max(total_iters - cfg.actor_lr_warmup, 1), 1.0)
                shape = min(trained / max(cfg.actor_lr_warmup, 1), 1.0) * (cfg.actor_lr_end + (1 - cfg.actor_lr_end) * 0.5 * (1 + np.cos(np.pi * past)))
                actor_lr, ff_lr = cfg.actor_lr_peak * shape, cfg.actor_lr_peak * shape
            actor_opt = agent.set_lr(actor_opt, actor_lr)
            if ff is not None:
                ff_opt = agent.set_lr(ff_opt, ff_lr)
            critic_opt = agent.set_lr(critic_opt, cfg.critic_lr * lr_scale)
            wm_opt = agent.set_lr(wm_opt, cfg.wm_lr * lr_scale)

            tic = time.time()
            for _ in range(cfg.collect_steps_per_iter):
                if ff is not None:
                    collector.step(lambda o: fns["policy_ff"](actor_params, batch_stats, ff, o, jnp.array(rd.timestep())), cfg.action_noise)
                else:
                    collector.step(collection_policy(actor_params, batch_stats), cfg.action_noise)
            timing["collect"] += time.time() - tic
            tic = time.time()
            for _ in range(cfg.wm_updates_per_iter):
                akey, sub = jax.random.split(akey)
                wm_params, wm_opt, wm_stats = wm_update(wm_params, wm_opt, collector.buf, collector.g_lo, collector.g_hi, sub)
            jax.block_until_ready(wm_params)
            timing["world_model"] += time.time() - tic
            tic = time.time()
            akey, sk, ck = jax.random.split(akey, 3)
            seeds = seeds_fn(collector.buf, collector.g_lo, collector.g_hi, sk)
            seeds.pop("valid")
            if ff is not None:
                new_ff, new_opt, traj, a_stats = fns["ff_update"](
                    ff, ff_opt, actor_params, batch_stats, wm_params, critic_params, seeds, voc_now, horizon=horizon
                )
                if it >= cfg.critic_warmup_iters:
                    ff, ff_opt = new_ff, new_opt
                    ema_ff = ema_update(ema_ff, ff)
            else:
                new_actor, new_opt, traj, a_stats = fns["actor_update"](
                    actor_params, actor_opt, batch_stats, wm_params, critic_params, seeds, voc_now, anchor_params, horizon=horizon
                )
                if it >= cfg.critic_warmup_iters:  # critic-only warmup: the bootstrap value starts random
                    actor_params, actor_opt = new_actor, new_opt
                    if cfg.policy_ema_decay > 0:
                        ema_params = ema_update(ema_params, actor_params)
            jax.block_until_ready(traj)
            timing["actor"] += time.time() - tic
            tic = time.time()
            critic_params, critic_opt, c_stats = fns["critic_update"](critic_params, critic_opt, traj, ck)
            jax.block_until_ready(critic_params)
            timing["critic"] += time.time() - tic
            it += 1
            n_loop += 1

            if cfg.voc_adaptive and it == cfg.critic_warmup_iters:
                collector.pop_perf()  # gate on the moving actor only
            if cfg.voc_adaptive and it > cfg.critic_warmup_iters and it % cfg.voc_check_iters == 0:
                perf = collector.pop_perf()
                enough = perf["frame0_episodes"] >= 20
                passed = (enough and perf["len_frac"] >= gate_len
                          and perf["carry"] >= cfg.voc_gate_carry and perf["touch"] >= cfg.voc_gate_touch)
                raised = False
                if passed:
                    voc_level = voc_level * cfg.voc_decay_factor
                    voc_level = 0.0 if voc_level < 0.03 else voc_level
                    gate_fails = 0
                else:
                    # DexMachina only lowers the gains; also give one step back when the policy keeps failing
                    # before the lift, so training stays at the hardest level it can still learn from
                    gate_fails = gate_fails + 1 if enough and perf["len_mean"] < collector.lift0 + 10 else 0
                    if cfg.voc_recover_patience > 0 and gate_fails >= cfg.voc_recover_patience:
                        voc_level = min(1.0, max(voc_level, 0.03) / cfg.voc_decay_factor)
                        gate_fails, raised = 0, True
                log("voc_check", iter=it, frac=frac, passed=passed, raised=raised, level=voc_level, gate_len=gate_len, **perf)

            if it % cfg.log_every_iters == 0:
                stats = jax.device_get({**wm_stats, **a_stats, **c_stats})
                n_t = cfg.log_every_iters
                log("train", iter=it, hours=elapsed / 3600, frac=frac, horizon=horizon, voc=voc_now, lr_scale=lr_scale,
                    actor_frozen=it <= cfg.critic_warmup_iters, env_steps=collector.g * W,
                    it_per_s=it / max(elapsed, 1e-6), **{f"sec_{n}": v / n_t for n, v in timing.items()},
                    gpu_mem_gb=jax.local_devices()[0].memory_stats().get("bytes_in_use", 0) / 1e9,
                    **collector.pop_window(), **{n: float(v) for n, v in stats.items()})
                timing = dict.fromkeys(timing, 0.0)
            if it % 500 == 0:
                akey, sub = jax.random.split(akey)
                log("wm_openloop", iter=it, **jax.device_get(wm_openloop(wm_params, collector.buf, collector.g_lo, collector.g_hi, sub)))
                # return-based reset bins: mean critic value per 1 s bin over recent settled rows
                akey, sub = jax.random.split(akey)
                sd = seeds_fn(collector.buf, collector.g_lo, collector.g_hi, sub)
                S = buffer.unpack_raw(sd["s"])
                feats = core.critic_features(k, S, sd["frame"], sd["estep"], jnp.full((cfg.imag_batch,), voc_now),
                                             core.lag_onehot(sd["lags"]))
                v = np.asarray(fns["critic_value"](critic_params, feats))
                b = np.asarray(sd["frame"]) // cfg.reset_bin_steps
                vals = np.array([v[b == i].mean() if (b == i).any() else collector.bin_value[i] for i in range(collector.nbins)])
                collector.update_bins(vals)
            if not cfg.final_only and (n_loop % cfg.export_every_iters == 0 if cfg.export_every_iters > 0
                    else time.time() - last_export >= cfg.export_every_minutes * 60):
                last_export = time.time()
                if ff is not None:
                    save_ff(it)
                    continue
                res = export.export(out / "checkpoints" / f"policy_{collector.g * W}.safetensors", actor_params, batch_stats,
                                    meta, training, collector.g * W, jnp.array(rd.observation())[:256], fns["policy_mean"])
                log("export", iter=it, **res)
                if res["fp16_max_abs_diff"] > 0.05:
                    log("warning", message="fp16 evaluator action differs from fp32 training action", diff=res["fp16_max_abs_diff"])
                exports.append(res["path"])
                if cfg.policy_ema_decay > 0:
                    export_ema(collector.g * W)
                if cfg.export_eval:
                    state = dict(actor_params=actor_params, batch_stats=batch_stats, actor_opt=actor_opt,
                                 critic_params=critic_params, critic_opt=critic_opt, wm_params=wm_params, wm_opt=wm_opt,
                                 wm_norm=wm_norm, critic_norm=critic_norm, bin_value=collector.bin_value,
                                 bin_hist=collector.bin_hist, frac=frac, voc_level=voc_level)
                    with open(out / "checkpoints" / f"state_{collector.g * W}.pkl", "wb") as fh:
                        pickle.dump(jax.device_get(state), fh)
                    evals = [p for p in evals if p.poll() is None]
                    evals.append(launch_eval(Path(res["path"]), out / "eval" / Path(res["path"]).stem, gpu, log, cfg.episode_index))

        if ff is not None:
            save_ff("final")
        res = export.export(out / "checkpoints" / f"policy_final_{collector.g * W}.safetensors", actor_params, batch_stats,
                            meta, training, collector.g * W, jnp.array(rd.observation())[:256], fns["policy_mean"])
        log("export", final=True, **res)
        if cfg.policy_ema_decay > 0:
            export_ema(f"final_{collector.g * W}")
        if not cfg.final_only:
            state = dict(actor_params=actor_params, batch_stats=batch_stats, actor_opt=actor_opt,
                         critic_params=critic_params, critic_opt=critic_opt, wm_params=wm_params, wm_opt=wm_opt,
                         wm_norm=wm_norm, critic_norm=critic_norm, bin_value=collector.bin_value,
                         bin_hist=collector.bin_hist, frac=min(frac, 1.0), voc_level=voc_level)
            with open(out / "checkpoints" / "state_final.pkl", "wb") as fh:
                pickle.dump(jax.device_get(state), fh)
        proc = launch_eval(Path(res["path"]), out / "eval" / "final", gpu, log, cfg.episode_index)
        for p in evals + [proc]:
            p.wait()
        log("done", iterations=it, env_steps=collector.g * W)


if __name__ == "__main__":
    sys.exit(main())
