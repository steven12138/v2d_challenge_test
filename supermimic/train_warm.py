"""FlashSAC warm start from a trained policy, without touching flash_chord (PLAN.md 13.22.3).

Wraps scripts/train_flash_sac.py: after the learner is created, copies the actor of a policy checkpoint into it and moves the step counter to
the curriculum stage to start from. A random critic would ruin a good actor, so run this with `training.optimization.actor_learning_rate_multiplier=0`
first (the critic learns the values of the warm policy), then continue with the official resume at the normal multiplier.
usage: WARM_POLICY=<policy.safetensors> WARM_STEPS=<start steps> python -m supermimic.train_warm <hydra overrides of train_flash_sac.py>
"""

import os
import runpy
import sys

import jax
import jax.numpy as jnp

import flash_chord.training.flash_sac.runner as runner
from flash_chord.training.flash_sac.checkpoint import ActorInferenceState, load_actor_checkpoint

POLICY = os.environ["WARM_POLICY"]
STEPS = int(os.environ["WARM_STEPS"])
create_learner = runner.create_learner


def warm_learner(*args, **kwargs):
    learner = create_learner(*args, **kwargs)
    template = ActorInferenceState(params=learner.actor.params, batch_stats=learner.actor.batch_stats)
    loaded, _ = load_actor_checkpoint(POLICY, template)
    put = lambda new, old: jax.device_put(jnp.asarray(new, old.dtype), next(iter(old.devices())))  # noqa: E731
    actor = learner.actor.replace(params=jax.tree.map(put, loaded.params, template.params),
                                  batch_stats=jax.tree.map(put, loaded.batch_stats, template.batch_stats))
    print(f"warm start: actor of {POLICY}, environment steps {STEPS}", flush=True)
    return learner.replace(actor=actor, environment_steps=learner.environment_steps + STEPS - int(learner.environment_steps))


runner.create_learner = warm_learner
sys.argv = ["train_flash_sac.py", *sys.argv[1:]]
sys.path.insert(0, "scripts")
runpy.run_path("scripts/train_flash_sac.py", run_name="__main__")
