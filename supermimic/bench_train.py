"""Benchmark of the training kernels on the real code path, variants timed in alternation to cancel GPU contention.

usage: python -m supermimic.bench_train <parquet> [rounds]    (XLA_FLAGS can be set per process to compare compiler flags)
"""

import os
import sys
import time
from dataclasses import replace

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import warp as wp
from functools import partial

from flash_chord.training.flash_sac.evaluation import compile_policy_action, create_actor_template
from flash_chord.training.flash_sac.checkpoint import ActorInferenceState
from flash_chord.training.flash_sac.network import Actor

from supermimic import agent, buffer, core, env_io, sonic_jax, world_model
from supermimic.config import Config
from supermimic.main import Collector, make_samplers


def timed(fn, rounds, warm=2):
    for _ in range(warm):
        jax.block_until_ready(fn())
    t = []
    for _ in range(rounds):
        tic = time.perf_counter()
        jax.block_until_ready(fn())
        t.append(time.perf_counter() - tic)
    return t


def main(parquet, rounds=12):
    W, H = 1024, 32
    cfg = Config(parquet=parquet, world_count=W, buffer_steps=300, imag_batch=6144, wm_batch=2048, wm_object_weight=5.0,
                 sat_penalty=0.5, frame0_seed_fraction=0.2, settle_reset_windows=True, reset_bin_steps=25,
                 w_object_velocity=2.0, w_grasp_contact=3.0, w_finger_grip=5.0)
    log = lambda *a, **k: None  # noqa: E731
    print("devices", jax.devices(), "jax", jax.__version__, "XLA_FLAGS", os.environ.get("XLA_FLAGS"), flush=True)
    with wp.ScopedDevice("cuda:0"):
        handles = env_io.build(parquet, W)
        env = handles.env
        consts = env_io.extract_consts(handles)
        consts.latent_table = sonic_jax.latent_table(env)
        k = core.device_consts(consts)
        k["delta_w"] = core.delta_weights(cfg.wm_object_weight)
        k["grip_q"] = jnp.asarray(core.finger_grip_table(consts, cfg.finger_grip_frac), jnp.float32)
        k["leg_dims"], k["arm_dims"] = core.residual_groups(consts)
        dec = sonic_jax.load_decoder_params()
        rd = env_io.Readout(env, consts)
        training = handles.training
        actor32 = Actor(action_dim=43, config=training.network, dtype=jnp.float32)
        actor16, template = create_actor_template(training, 486, 43)
        key = jax.random.PRNGKey(0)
        policy16 = compile_policy_action(actor16, template, jnp.zeros((W, 486)), key, deterministic=True,
                                         action_transform="tanh", action_scale=10.0)
        state = ActorInferenceState(params=template.params, batch_stats=template.batch_stats)
        policy = lambda o: policy16(state, o, key)[0]  # noqa: E731
        env.set_voc_scale(1.0)
        c = Collector(cfg, handles, k, rd, log)
        tic = time.perf_counter()
        for _ in range(200):
            c.step(policy, cfg.action_noise)
        col = (time.perf_counter() - tic) / 200
        zero = wp.zeros(env.action_input.shape, dtype=wp.float32)
        tic = time.perf_counter()
        for _ in range(50):
            env.step(zero)
            wp.synchronize_device()
        sim = (time.perf_counter() - tic) / 50
        print(f"collector.step {1000 * col:.1f} ms of which env.step {1000 * sim:.1f} ms ({W} worlds)", flush=True)
        import cProfile
        import pstats

        prof = cProfile.Profile()
        prof.enable()
        for _ in range(100):
            c.step(policy, cfg.action_noise)
        prof.disable()
        pstats.Stats(prof).sort_stats("tottime").print_stats(22)

        sample_windows, sample_seeds = make_samplers(k, cfg, W, cfg.buffer_steps)
        r0 = {n: v[0] for n, v in jax.jit(partial(sample_windows, length=1, batch=8192))(c.buf, c.g_lo, c.g_hi, key).items()}
        S, Sn = buffer.unpack_raw(r0["s"]), buffer.unpack_raw(r0["sn"])
        prev = jnp.where((r0["estep"] > 0)[:, None], r0["prev_target"], r0["target"])
        m = r0["valid"] > 0
        wm_norm = {"in": world_model.normalizer_from(core.wm_inputs(k, S, r0["target"], prev, core.lag_onehot(r0["lags"]), r0["voc"], r0["frame"])[m]),
                   "out": world_model.normalizer_from(core.encode_delta(S, Sn)[m], floor=1e-4)}
        critic_norm = world_model.normalizer_from(core.critic_features(k, S, r0["frame"], r0["estep"], r0["voc"], core.lag_onehot(r0["lags"]))[m])
        wm_params = world_model.init(jax.random.PRNGKey(11), cfg.wm_members, cfg.wm_hidden, cfg.wm_layers)
        wm_tx = optax.chain(optax.clip_by_global_norm(1.0), optax.inject_hyperparams(optax.adam)(learning_rate=cfg.wm_lr))
        wm_opt = wm_tx.init(wm_params)
        weights = jnp.asarray(env.objective.weights.numpy())
        extra = jnp.ones(len(core.EXTRA_NAMES))
        actor_params, batch_stats = template.params, template.batch_stats

        def wm_one(params, opt, buf, lo, hi, key):
            keys = jax.random.split(key, cfg.wm_members)
            rows = jax.vmap(lambda kk: sample_windows(buf, lo, hi, kk, cfg.wm_window, cfg.wm_batch))(keys)
            loss_fn = lambda p: jnp.sum(jax.vmap(lambda q, r: world_model.window_loss(q, k, wm_norm, r)[0])(p, rows))  # noqa: E731
            g = jax.grad(loss_fn)(params)
            u, opt = wm_tx.update(g, opt, params)
            return optax.apply_updates(params, u), opt

        wm_jit = jax.jit(wm_one)

        @jax.jit
        def wm_scan8(params, opt, buf, lo, hi, key):
            def body(carry, kk):
                return wm_one(*carry, buf, lo, hi, kk), None
            return jax.lax.scan(body, (params, opt), jax.random.split(key, 8))[0]

        def wm_loop8():
            p, o = wm_params, wm_opt
            for kk in jax.random.split(key, 8):
                p, o = wm_jit(p, o, c.buf, c.g_lo, c.g_hi, kk)
            return p

        seeds_fn = jax.jit(partial(sample_seeds, n=cfg.imag_batch, frame0_frac=cfg.frame0_seed_fraction))
        seeds = seeds_fn(c.buf, c.g_lo, c.g_hi, key)
        seeds.pop("valid")

        def actor_variant(**kw):
            fns = agent.make_fns(k, replace(cfg, **kw), actor32, dec, wm_norm, critic_norm, weights, extra)
            opt = fns["actor_tx"].init(actor_params)
            critic = agent.critic_init(jax.random.PRNGKey(13), cfg.critic_hidden)
            return (lambda: fns["actor_update"](actor_params, opt, batch_stats, wm_params, critic, seeds, 1.0, horizon=H)[0]), fns, critic

        tests = {}
        tests["seed sampling"] = lambda: seeds_fn(c.buf, c.g_lo, c.g_hi, key)["s"]
        tests["wm 8 x jit"] = wm_loop8
        tests["wm scan 8"] = lambda: wm_scan8(wm_params, wm_opt, c.buf, c.g_lo, c.g_hi, key)[0]
        tests["actor base"], fns0, critic0 = actor_variant()
        tests["actor sonic stopgrad"] = actor_variant(sonic_stop_grad=True)[0]
        for name in ("tensorfloat32", "float32"):
            with jax.default_matmul_precision(name):
                tests[f"actor base [{name}]"] = actor_variant()[0]
        traj = fns0["actor_update"](actor_params, fns0["actor_tx"].init(actor_params), batch_stats, wm_params, critic0, seeds, 1.0, horizon=H)[2]
        copt = fns0["critic_tx"].init(critic0)
        tests["critic"] = lambda: fns0["critic_update"](critic0, copt, traj, key)[0]

        names = list(tests)
        for n in names:  # compile (context managers above only select the precision at trace time)
            print("compile", n, flush=True)
            prec = n.split("[")[1][:-1] if "[" in n else None
            if prec:
                with jax.default_matmul_precision(prec):
                    jax.block_until_ready(tests[n]())
            else:
                jax.block_until_ready(tests[n]())
        res = {n: [] for n in names}
        for r in range(rounds):
            for n in names:
                prec = n.split("[")[1][:-1] if "[" in n else None
                tic = time.perf_counter()
                if prec:
                    with jax.default_matmul_precision(prec):
                        jax.block_until_ready(tests[n]())
                else:
                    jax.block_until_ready(tests[n]())
                res[n].append(time.perf_counter() - tic)
        base = np.median(res["actor base"])
        for n in names:
            print(f"{n:36s} median {1000 * np.median(res[n]):8.1f} ms  min {1000 * np.min(res[n]):8.1f} ms", flush=True)
        print("memory GB", jax.local_devices()[0].memory_stats().get("peak_bytes_in_use", 0) / 1e9, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12)
