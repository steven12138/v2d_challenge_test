"""FlashSAC training with a guard against diverged worlds, without touching flash_chord (PLAN.md 13.22.7).

One world whose contact solver returns NaN is enough to end a run: its observation and reward enter the replay buffer and the next updates turn every actor and
critic weight into NaN (ep 7 at 271M steps, at 94 % progress, and again 30M steps into a resume). A reset does not bring such a world back, because the NaN stays
in the solver's internal state, but it does not reach the other worlds. So a diverged world is quarantined: from then on its rows are copies of a healthy world's
observation with zero reward and a finished episode. Losing a few of 2048 worlds costs little.
usage: python -m supermimic.train_guard <hydra overrides of train_flash_sac.py>     run from the flash_chord directory
"""

import runpy
import sys

import jax.numpy as jnp
import numpy as np
import warp as wp

from flash_chord.envs.rl import RLEnv
from flash_chord.runtime.jax import to_jax

MAX_DEAD_FRACTION = 0.1
original_step = RLEnv.step


def patch_rows(array, dead, source, width):
    """Rows of the dead worlds in a flat Warp buffer become copies of the row of a healthy world."""
    values = array.numpy().reshape(-1, width).copy()
    values[dead] = values[source]
    array.assign(values.reshape(-1))


def guarded_step(self, action=None):
    result = original_step(self, action)
    if not hasattr(self, "_guard_observation"):
        self._guard_dim = self.observation.shape[0] // self.world_count
        self._guard_observation = to_jax(self.observation, (self.world_count, self._guard_dim))
        self._guard_reward = to_jax(self.reward)
        self._guard_dead = np.zeros(self.world_count, bool)
    wp.synchronize_device(self.device)
    if bool(jnp.isfinite(jnp.sum(self._guard_observation) + jnp.sum(self._guard_reward))):
        return result
    observation = self.observation.numpy().reshape(self.world_count, self._guard_dim)
    reward = self.reward.numpy()
    new = (~np.isfinite(observation).all(1) | ~np.isfinite(reward)) & ~self._guard_dead
    self._guard_dead |= new
    dead = self._guard_dead
    if new.any():
        print(f"guard: quarantined {int(new.sum())} diverged worlds, {int(dead.sum())} of {self.world_count} in total", flush=True)
    if dead.mean() > MAX_DEAD_FRACTION:
        raise RuntimeError(f"{int(dead.sum())} of {self.world_count} worlds diverged")
    source = int(np.flatnonzero(~dead)[0])
    patch_rows(self.observation, dead, source, self._guard_dim)
    for name in ("terminal_observation", "critic_context", "terminal_critic_context"):
        buffer = getattr(self, name, None)
        if buffer is not None:
            patch_rows(buffer, dead, source, buffer.shape[0] // self.world_count)
    terms = getattr(getattr(self, "objective", None), "terms", None)
    if terms is not None:
        patch_rows(terms, dead, source, terms.shape[0] // self.world_count)
    reward[dead] = 0.0
    self.reward.assign(reward)
    done = self.done.numpy()
    done[dead] = 1
    self.done.assign(done)
    wp.synchronize_device(self.device)
    return result


RLEnv.step = guarded_step

if __name__ == "__main__":
    sys.argv = ["train_flash_sac.py", *sys.argv[1:]]
    sys.path.insert(0, "scripts")
    runpy.run_path("scripts/train_flash_sac.py", run_name="__main__")
