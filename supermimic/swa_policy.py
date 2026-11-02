"""Average the weights of several policies of one run (stochastic weight averaging, no training).

FlashSAC policies swing between checkpoints and the EMA helped the SuperMimic refinement a lot. Averages parameters and normalization
statistics, projects back onto the unit constraints the actor trains under, and keeps the metadata of the last input.
usage: python -m supermimic.swa_policy <out.safetensors> <policy.safetensors> [<policy.safetensors> ...]
"""

import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from flash_chord.configuration import instantiate_typed
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import checkpoint_config
from flash_chord.training.flash_sac.checkpoint import ActorInferenceState, load_actor_checkpoint, save_actor_checkpoint
from flash_chord.training.flash_sac.config import TrainingConfig
from flash_chord.training.flash_sac.evaluation import create_actor_template
from flash_chord.training.flash_sac.network import project_unit_parameters

FILE_KEYS = ("format", "manifest")


def main(out: str, policies: list[str]) -> None:
    metadata = read_checkpoint_metadata(policies[-1])
    training = instantiate_typed(OmegaConf.create(checkpoint_config(metadata)).training, TrainingConfig)
    _, template = create_actor_template(training, 486, 43)
    states = [load_actor_checkpoint(p, template)[0] for p in policies]
    mean = jax.tree.map(lambda *leaves: jnp.mean(jnp.stack(leaves), axis=0), *states)
    state = ActorInferenceState(params=project_unit_parameters(mean.params), batch_stats=mean.batch_stats)
    save_actor_checkpoint(out, state, metadata={k: v for k, v in metadata.items() if k not in FILE_KEYS})
    print(f"{out}: average of {len(policies)} policies", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
