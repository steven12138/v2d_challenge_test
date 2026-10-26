# Track 2 baseline: architecture

Goal: reproduce the official Track 2 baseline for **Tier 1** (clean) and **Tier 2** (synthetic noise)
without the official Docker tooling. The algorithms are the official ones, copied into this repository
so they can be edited. Each stage is one script that takes explicit input and output paths, so any
data in the same format (e.g. our own reconstructions) can enter at any stage.

## Pipeline

```
LeRobot dataset (tier_1_multiview_caption | tier_2_synthetic_noise | external data)
   │  prep/prep.py               ours: parquet -> SOMA folders + support stages
   ▼
<soma_dir>/sequence_id=<seq>/    soma_params.npz, poses.npy, ground_plane.json, reconstructed_mesh/
   │  retarget/soma_to_g1.py     official SOMA-X -> G1 retarget (Pink IK), CPU
   ▼
<motion_root>/whole_body/soma/sequence_id=<seq>/robot_name=g1/data.parquet
<motion_root>/whole_body/reconstructed_stage/<seq>_support.usda   (written by prep)
   │  train/train.py             official FlashCHORD trainer, recipe g1_recon_body_flash_sac, 1x A40
   ▼
<train_dir>/<seq>/policy_349999104.safetensors
   │  evaluate/checkpoints.py    ours: episode index -> policy map that the kit takes
   │  official submission kit    run_policy_evaluation.py -> pack_submission.py -> v2d_submit.py
   ▼
submission CSV + the four local scores (AUC, SP-SR, MP-SR, MPPE)
```

Each episode gets its own policy, so a tier needs 30 policies. All tiers are scored against the clean
Tier 1 object trajectory. The kit packer maps each rollout into the released world frame with one
rigid transform, derived from that tier's input reference.

## Repository layout

| Path | Contents |
|---|---|
| `prep/prep.py` | Our converter from LeRobot Track 2 data to the retargeter's input. |
| `retarget/` | Official `soma_to_g1.py` and `setup_soma_assets.py`, plus the `robotic_grounding` subset they import: `retarget/`, `motion_schema/`, and the G1 URDF and meshes. The only change: one `sys.path` line so the scripts import this local copy. |
| `train/train.py` | Copy of the official `train_flash_sac.py`. It imports `train/flash_chord/src`, so edits to FlashCHORD modules take effect directly. |
| `train/flash_chord/` | Official FlashCHORD v0.4, unmodified (without its docs, tests, docker and workflow folders). |
| `evaluate/checkpoints.py` | Writes the `{episode_index: checkpoint}` JSON the kit evaluator needs. |
| `external/video_to_data/` | Git clone of the official repo at kit evaluator commit `8c778f9`. Reference only; nothing imports it. |
| `external/v2d_submission_kit/` | Official submission kit from the challenge site (`KIT_SOURCE.txt`). Used unmodified for evaluation, packing and upload. |
| `slurm/` | One sbatch script per GPU/CPU stage; paths are positional arguments. |
| `visualize/` | Local MP4s: `render.sh` (retargeted G1 kinematic reference + zero-residual SONIC replay) and `render_input.py` (SOMA-X reconstruction in the same G1 world). |
| `scripts/` | `sync_to_grace.sh` and the one-time `setup_grace.sh`. |

Both copied folders record the upstream commit in `UPSTREAM_COMMIT`.

## Environments on Grace

Two existing uv envs under `/scratch/user/the/v2d/envs/`, pinned to the official lockfile/Dockerfile:

| Env | Stages | Key pins |
|---|---|---|
| `rg` | prep, retarget | py-soma-x 0.2.1, pin 3.7.0, pin-pink 4.2.0, torch 2.5.1 |
| `flash` | train, evaluate, pack | jax 0.10.2 (CUDA 13), newton 1.2.0, warp 1.14.0, mujoco 3.8.1; `train/flash_chord` installed editable, so the kit's evaluator imports the same code we train with |

`slurm/env.sh` adds the env's `nvidia/cu13/lib` and `nvidia/cudnn/lib` to `LD_LIBRARY_PATH`.
onnxruntime-gpu (the SONIC encoder/decoder) needs CUDA 13 + cuDNN 9, which the official image ships
system-wide. `nvidia-curand==10.4.0.35` was added to `flash` for the same reason.

GPU memory: the recipe needs 48 GB (A40). About 21 GiB of that is the replay ring: 10M transitions x
~2.3 KB (486-dim float16 obs + next obs, plus float32 action, objective terms and critic context).
`slurm/train_a100.sbatch` sets `training.replay.capacity=5000000` to fit a 40 GB A100.

## Conventions that must hold (checked against the official code)

- Use `<motion_root> = <anything>/assets/human_motion_data`, as in the official repo. soma_to_g1.py stores
  absolute object-mesh paths in the parquet; FlashCHORD re-roots any path containing `/assets/` onto the
  parquet's nearest `assets` ancestor, so a motion directory copied elsewhere (Grace <-> local) still loads.
- `poses.npy` holds `world_T_object` in the **OpenCV** world. `soma_to_g1.py` converts it to the SOMA
  axes, levels the fitted ground plane and anchors everything at frame 0. The support stages are
  already expressed in that `g1_retarget_world` frame.
- The kit packer requires the evaluator's reference object poses to match the released input up to
  **one rigid transform (within 1 mm / 1 mrad)**. So retargeting must not alter the object trajectory
  beyond that transform; the official ground-lift is a constant offset, which still satisfies it.
- 16 of the 30 sequences are ground-only, and their released support stage is empty. FlashCHORD
  raises an error on an empty stage, so train (and render) them with `scene.support=false`, as the
  official blue-trash-can example does. The setting is saved in the checkpoint, so the evaluator
  rebuilds the same scene.
- Only episode 11 has invisible frames (one frame). Prep fills them by interpolation; the packer
  drops invisible frames before its check.
- A submission must carry the hashes of the kit's `evaluate_policy.py` and of FlashCHORD's evaluation
  modules. Keep `train/flash_chord/src/flash_chord/evaluation/` unmodified so they match the official
  ones.
