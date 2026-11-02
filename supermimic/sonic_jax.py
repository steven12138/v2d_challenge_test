"""SONIC decoder weights (extracted from the pinned ONNX) and per-frame encoder latents."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets"


def load_decoder_params() -> dict:
    data = np.load(ASSETS / "sonic_decoder.npz")
    return {key: jnp.asarray(data[key]) for key in data.files}


def latent_table(env) -> np.ndarray:
    """Encoder output for every reference frame with zero reference offset.

    The encoder input depends only on (frame, offset) (sonic.py gather_sonic_encoder_input), and offsets
    are zero for every settled/explicit world, which are the only states the imagination starts from.
    """
    import onnxruntime as ort
    from flash_chord.runtime.actions.sonic import _POLICY_DIR

    encoder_reference = np.asarray(env.action.encoder_reference.numpy(), dtype=np.float32)
    session = ort.InferenceSession(
        str(_POLICY_DIR / "encoder_batched.onnx"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    out = session.run(["encoded_tokens"], {"obs_dict": encoder_reference})[0]
    return np.asarray(out, dtype=np.float32)
