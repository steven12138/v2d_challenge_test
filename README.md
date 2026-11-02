# Track 2 frozen recipe v1: Pink IK motion + object keypoint reward + weight averaging

One FlashSAC policy per episode, rolled out by the official FlashCHORD evaluator (runtime commit in `grace/train/UPSTREAM_COMMIT`).

## Recipe

1. **Motion:** the official Pink IK retargeted motion of the challenge dataset, unchanged.
2. **Training:** official FlashSAC recipe (`g1_recon_body_flash_sac`, 2048 worlds, 349,999,104 steps, same curriculum) plus one reward:
   `object_keypoints`, weight 3.0, std 0.1 m. It is a Gaussian on the mean squared distance of 6 points at +-object radius along the object axes
   (ResMimic, arXiv 2510.05070), so it covers rotation like the AUC and MPPE metrics. See `grace/flash_chord_keypoints.patch` against the
   runtime commit. Experiment: `g1_recon_body_kp_flash_sac`, seed 42. Job: `grace/slurm/train_a100.sbatch`, submitted by `scripts/submit_grace.sh`.
3. **Averaging:** `supermimic/swa_policy.py` averages the weights and normalization statistics of the last 3 policies (330M, 340M, 350M steps)
   and projects them back onto the unit constraints of the actor. No further training.
4. **Official config:** `supermimic/kp_checkpoint.py` removes the keypoint reward from the config stored in the checkpoint, so the unmodified
   evaluator can rebuild the environment. Weights are byte-identical. It runs on each input of step 3 before averaging.

## Evaluate

- `supermimic/diag_eval_stats.py`: 64 worlds under evaluation conditions, with the official AUC and MPPE functions.
- `supermimic/eval_episode.sh`: official kit evaluator and packer.

A single official run is not bitwise reproducible (GPU contact solving), so a policy is judged on repeats.

## Files

| Path | What |
|---|---|
| `scripts/episodes.txt` | episode, sequence, and the training output directory on Grace |
| `CHECKPOINTS.tsv` | episode, sha256, note for each policy in `checkpoints/` (not in git) |
| `RESULTS.md` | measured scores of this version |

Differences from the plain recipe by episode are listed in `RESULTS.md`.
