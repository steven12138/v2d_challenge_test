"""Export the zero-residual (pure SONIC) checkpoint of one episode for the official evaluator.

usage: python -m supermimic.sonic_baseline <parquet> <out.safetensors>
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import warp as wp

from flash_chord.training.flash_sac.evaluation import create_actor_template
from flash_chord.training.flash_sac.network import Actor

from supermimic import env_io, export


def main(parquet, out):
    with wp.ScopedDevice("cuda:0"):
        handles = env_io.build(parquet, 1)
        training = handles.training
        actor32 = Actor(action_dim=43, config=training.network, dtype=jnp.float32)
        _, template = create_actor_template(training, 486, 43)
        mean = jax.jit(lambda p, s, o: jnp.tanh(actor32.apply({"params": p, "batch_stats": s}, o, training=False).mean) * 10.0)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        print(export.export(Path(out), template.params, template.batch_stats, handles.checkpoint_metadata, training, 0,
                            jnp.zeros((64, 486)), mean), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
