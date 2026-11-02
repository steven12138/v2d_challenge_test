"""Write the trained actor as an official FlashSAC policy checkpoint and verify it round-trips."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from flash_chord.training.flash_sac.checkpoint import ActorInferenceState, load_actor_checkpoint, save_actor_checkpoint
from flash_chord.training.flash_sac.evaluation import compile_policy_action, create_actor_template


def export(path: Path, actor_params, batch_stats, metadata: dict, training, environment_steps: int, obs_sample, fp32_policy):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = ActorInferenceState(params=actor_params, batch_stats=batch_stats)
    meta = dict(metadata)
    meta.update(curriculum_stage=8, environment_steps=int(environment_steps), voc_scale=0.0)
    save_actor_checkpoint(path, state, metadata=meta)

    # Round trip through the evaluator's own loader and fp16 inference path.
    actor16, template = create_actor_template(training, obs_sample.shape[1], 43)
    t_leaves, t_def = jax.tree.flatten(template)
    o_leaves, o_def = jax.tree.flatten(state)
    if len(t_leaves) != len(o_leaves) or any(a.shape != b.shape for a, b in zip(t_leaves, o_leaves)):
        raise RuntimeError("actor state does not match the FlashSAC template")
    loaded, loaded_meta = load_actor_checkpoint(path, template)
    for a, b in zip(jax.tree.leaves(loaded), o_leaves):
        if not np.array_equal(np.asarray(a), np.asarray(b)):
            raise RuntimeError("checkpoint leaves changed on reload")
    key = jax.random.PRNGKey(0)
    policy16 = compile_policy_action(
        actor16, loaded, obs_sample, key, deterministic=True,
        action_transform=training.network.action_transform, action_scale=training.network.action_scale,
    )
    a16, _ = policy16(loaded, obs_sample, key)
    a32 = fp32_policy(actor_params, batch_stats, obs_sample)
    diff = float(jnp.max(jnp.abs(a16 - a32)))
    return dict(path=str(path), fp16_max_abs_diff=diff, curriculum_stage=loaded_meta.get("curriculum_stage"))
