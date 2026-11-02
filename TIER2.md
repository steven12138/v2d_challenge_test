# Track 2 Tier 2 recipe, version 1

One policy per episode, rolled out by the official FlashCHORD evaluator (runtime commit in `T2_INPUTS.sha256`). Every step reads only the released Tier 2 input and physical priors: no Tier 1 data enters the input, the training or the choice of policy.

## Recipe

1. **Input:** `supermimic/t2_anchor.py` removes the random walk of the released Tier 2 object poses. Rests are found in the input itself (`still_segments`) and settled onto a stable face of the object and a touched surface; the correction between rests is a Brownian bridge on the world side of the rotation, followed by a 5 Hz low-pass. The human motion gets a low-pass (`supermimic/t2_filter.py`).
2. **Retargeting:** `supermimic/t2_retarget.sh`, the official Pink IK path, on that input.
3. **Contact labels:** `supermimic/t2_contact.py` repairs the labels lost to the noise: a palm near the surface while the object moves. Result: motion root `t2b`, hashed in `T2_INPUTS.sha256`.
4. **Training:** official FlashSAC (`g1_recon_body_flash_sac`, 2048 worlds, 349,999,104 steps, seed 42) with the keypoint reward of `supermimic/flash_objective.py`, the NaN guard `supermimic/train_guard.py`, and `palm_orientation` weight 0 in all 9 curriculum stages, because the palm targets carry the noise of the human. `scripts/t2_batch.py` writes the queue lines, `supermimic/run_variant.sh` and `supermimic/run_flash.sh` run them. The last 3 policies are averaged (`supermimic/swa_policy.py`).
5. **Refinement:** `supermimic/run_t2_refine.sh`, two passes of SuperMimic (`world_count=4096 imag_batch=3072 w_anchor=5 actor_lr=3e-5 w_object_orientation=3 w_object_orientation_coarse=3`), each from the best policy so far.
6. **Choice:** `supermimic/t2_pick.py`, 64-world AUC against the policy input minus the fall rate. The local official score reads the clean reference, so it is only reported and never chooses anything.
7. **Official config:** `supermimic/kp_checkpoint.py` removes training-only settings from the config stored in the checkpoint.

## Packing

`supermimic/build_tier2.sh` gives each policy one fresh official run and packs the CSV with `--frame-references data/raw/tier_2_anchored`, since the policy input is the output of step 1 and not the released file. That reference is in the released Track 2 world frame. Poses are the achieved rollouts, and they are scored against the clean reference as usual.

## Files

| Path | What |
|---|---|
| `T2_INPUTS.sha256` | runtime commit, hash of the step 1 output, hash of each motion |
| `supermimic/` | all code, as it runs on the training machine; `supermimic/assets/` (SONIC decoder weights converted from the official ONNX) is not in git |
| `scripts/t2_batch.py` | queue lines of the training runs |
