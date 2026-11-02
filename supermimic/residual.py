"""External residual policy inside one official FlashSAC actor (PLAN.md 13.22.4).

The evaluator builds `Actor(config=training.network)` from the checkpoint config, so the network can be wider than the trained one. Every actor
layer works per feature except the final RMS norm, so a block-diagonal kernel holds two independent subnetworks: the frozen base policy and a new
residual branch whose mean head starts at zero. As in the downstream interfaces of the SuperMimic paper (appendix A), the return gradient passes
through the frozen part and the same world model, and only the new branch trains.

The final RMS norm averages over all features, so the branch is kept small (`epsilon`) to leave the base almost untouched; its scale undoes that.
Unit projections would break the block structure and are not applied (the evaluator does not apply them either).
usage: python -m supermimic.residual <base.safetensors> <out.safetensors> <branch width> [epsilon]   writes the zero-residual policy
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
from omegaconf import OmegaConf

from flash_chord.configuration import instantiate_typed
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import checkpoint_config
from flash_chord.training.flash_sac.checkpoint import ActorInferenceState, load_actor_checkpoint, save_actor_checkpoint
from flash_chord.training.flash_sac.config import TrainingConfig
from flash_chord.training.flash_sac.evaluation import create_actor_template

CONFIG_KEYS = ("resolved_config_json", "user.resolved_config_json")
EPSILON = 0.05


def widen_training(training, extra: int):
    """TrainingConfig with `extra` more actor features."""
    network = dataclasses.replace(training.network, actor_hidden_dim=training.network.actor_hidden_dim + extra)
    return dataclasses.replace(training, network=network)


def widen_metadata(metadata: dict, hidden: int) -> dict:
    """Checkpoint metadata whose stored config builds the wider actor."""
    meta = dict(metadata)
    for key in CONFIG_KEYS:
        if key in meta:
            raw = meta[key]
            config = json.loads(raw) if isinstance(raw, str) else raw
            config["training"]["network"]["actor_hidden_dim"] = hidden
            meta[key] = json.dumps(config, sort_keys=True) if isinstance(raw, str) else config
            return meta
    raise KeyError(f"no resolved config in the metadata keys {sorted(meta)}")


def _blockwise(base, new, rows, cols):
    """`base` on the top-left block, `new` on the bottom-right block, zero across."""
    wide = jnp.zeros_like(new).at[rows:, cols:].set(new[rows:, cols:])
    return wide.at[:rows, :cols].set(base)


def widen_params(base: dict, template: dict, h0: int, epsilon: float = EPSILON) -> dict:
    """Parameters of the wide actor from the base actor; `template` supplies the initial branch weights."""
    h = template["embed"]["unit_kernel"].shape[1]
    out = unfreeze(template)
    base = unfreeze(base)
    out["input_norm"] = base["input_norm"]
    out["embed"]["unit_kernel"] = template["embed"]["unit_kernel"].at[:, :h0].set(base["embed"]["unit_kernel"]).at[:, h0:].multiply(epsilon)
    expansion = template["block_0"]["expand"]["unit_kernel"].shape[1] // h
    for name in [n for n in base if n.startswith("block_")]:
        b, t = base[name], out[name]
        t["expand"]["unit_kernel"] = _blockwise(b["expand"]["unit_kernel"], t["expand"]["unit_kernel"], h0, expansion * h0)
        t["contract"]["unit_kernel"] = _blockwise(b["contract"]["unit_kernel"], t["contract"]["unit_kernel"], expansion * h0, h0)
        for norm, n0 in (("expand_norm", expansion * h0), ("contract_norm", h0)):
            t[norm]["unit_scale"] = t[norm]["unit_scale"].at[:n0].set(b[norm]["unit_scale"])
            t[norm]["unit_bias"] = t[norm]["unit_bias"].at[:n0].set(b[norm]["unit_bias"])
        t["contract_norm"]["unit_scale"] = t["contract_norm"]["unit_scale"].at[h0:].set(epsilon)
    shrink = math.sqrt(h0 / h)  # with a small branch the RMS over h features is the base RMS times this
    scale = jnp.full((h,), shrink / epsilon)
    out["post_norm"]["rms_scale"] = scale.at[:h0].set(base["post_norm"]["rms_scale"] * shrink)
    out["mean"]["kernel"] = jnp.zeros_like(template["mean"]["kernel"]).at[:h0].set(base["mean"]["kernel"])
    out["log_std"]["unit_kernel"] = jnp.zeros_like(template["log_std"]["unit_kernel"]).at[:h0].set(base["log_std"]["unit_kernel"])
    out["log_std"]["gain"] = base["log_std"]["gain"]
    out["mean_bias"] = base["mean_bias"]
    out["log_std_bias"] = base["log_std_bias"]
    return freeze(out) if not isinstance(template, dict) else out


def widen_stats(base: dict, template: dict, h0: int) -> dict:
    """Normalization statistics: the base's, and unit statistics on the branch until `calibrate`."""
    out = unfreeze(template)
    base = unfreeze(base)
    out["input_norm"] = base["input_norm"]
    for name in [n for n in base if n.startswith("block_")]:
        for norm in ("expand_norm", "contract_norm"):
            for stat in ("running_mean", "running_variance"):
                n0 = base[name][norm][stat].shape[0]
                out[name][norm][stat] = out[name][norm][stat].at[:n0].set(base[name][norm][stat])
    return freeze(out) if not isinstance(template, dict) else out


def trainable_mask(params: dict, h0: int):
    """1 on the entries of the residual branch, 0 on the frozen base and on the fixed branch amplitude."""
    expansion = params["block_0"]["expand"]["unit_kernel"].shape[1] // params["embed"]["unit_kernel"].shape[1]

    def leaf(path, x):
        name = jax.tree_util.keystr(path)
        m = jnp.zeros_like(x)
        if "['embed']" in name:
            return m.at[:, h0:].set(1.0)
        if "['expand']" in name:
            return m.at[h0:, expansion * h0:].set(1.0)
        if "['contract']" in name:
            return m.at[expansion * h0:, h0:].set(1.0)
        if "['expand_norm']" in name:
            return m.at[expansion * h0:].set(1.0)
        if "['mean']" in name:
            return m.at[h0:].set(1.0)
        if "['mean_bias']" in name:
            return jnp.ones_like(x)
        return m

    return jax.tree_util.tree_map_with_path(leaf, params)


def calibrate(actor, params, stats, observations, h0: int, passes: int = 500):
    """Normalization statistics of the branch from real observations; the base's stay exactly as trained."""
    @jax.jit
    def step(s):
        _, changed = actor.apply({"params": params, "batch_stats": s}, observations, training=True, mutable=["batch_stats"])
        return changed["batch_stats"]

    current = stats
    for _ in range(passes):
        current = step(current)
    expansion = params["block_0"]["expand"]["unit_kernel"].shape[1] // params["embed"]["unit_kernel"].shape[1]
    out = unfreeze(current)
    frozen = unfreeze(stats)
    out["input_norm"] = frozen["input_norm"]
    for name in [n for n in out if n.startswith("block_")]:
        for norm, n0 in (("expand_norm", expansion * h0), ("contract_norm", h0)):
            for stat in ("running_mean", "running_variance"):
                out[name][norm][stat] = out[name][norm][stat].at[:n0].set(frozen[name][norm][stat][:n0])
    return freeze(out) if not isinstance(stats, dict) else out


def widen_policy(base_path: str, extra: int, epsilon: float = EPSILON):
    """(wide state, wide training, base state, base training, metadata) of a policy file."""
    metadata = read_checkpoint_metadata(base_path)
    training = instantiate_typed(OmegaConf.create(checkpoint_config(metadata)).training, TrainingConfig)
    _, base_template = create_actor_template(training, 486, 43)
    base, _ = load_actor_checkpoint(base_path, base_template)
    wide_training = widen_training(training, extra)
    _, wide_template = create_actor_template(wide_training, 486, 43)
    h0 = training.network.actor_hidden_dim
    state = ActorInferenceState(params=widen_params(base.params, wide_template.params, h0, epsilon),
                                batch_stats=widen_stats(base.batch_stats, wide_template.batch_stats, h0))
    return state, wide_training, base, training, widen_metadata(metadata, h0 + extra)


if __name__ == "__main__":
    wide, wide_training, _, _, meta = widen_policy(sys.argv[1], int(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else EPSILON)
    save_actor_checkpoint(sys.argv[2], wide, metadata={k: v for k, v in meta.items() if k not in ("format", "manifest")})
    print(f"{sys.argv[2]}: actor width {wide_training.network.actor_hidden_dim}, zero residual", flush=True)
