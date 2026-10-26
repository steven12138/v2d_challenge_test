# Improving retargeting and RL: plan

Scope: replace or upgrade the two learned stages of the baseline (retargeting, RL) with more recent methods, first
on the two sequences we already trained:

| Sequence | Baseline result (official recipe, 350M steps, 1-world kit evaluator) |
|---|---|
| `round_table_flip_08` (ground only) | MPPE 3.4 cm, pos 3.3 cm, ori 11.2°, AUC 0.41, SP-SR 1, MP-SR 0 (A40) |
| `snack_box_desk_through_06` (desk support) | MPPE 28.2 cm, AUC 0.24, SP-SR 0: drops the box when it passes between the feet, terminates, restarts (A100) |

Hard constraint: a submission must be rolled out by the official FlashCHORD evaluator, which rebuilds the env from
the checkpoint config. So the policy stays a FlashCHORD (FlashSAC) checkpoint and
`flash_chord/evaluation/` stays unmodified. Rewards, termination, curriculum and the reference motion are free to change.

## What the literature suggests

- **ResMimic** (arXiv 2510.05070) trains a residual policy on top of a general motion-tracking prior, the same
  structure as FlashCHORD with SONIC. Its object reward is a **point-cloud distance** (sampled mesh points),
  which covers translation and rotation together, plus a link-level **contact reward** and early termination
  on point-cloud deviation. Ablation: 92.5% success vs 7.5% for fine-tuning.
- **HDMI** (arXiv 2509.16757): co-tracks robot and object, uses a residual action space and a general
  interaction reward.
- **OmniRetarget** (arXiv 2509.26633; code in `amazon-far/holosoma`, `holosoma_retargeting`) does
  interaction-mesh retargeting. It preserves the spatial relations between the body and object surface points
  under hard constraints: non-penetration, foot sticking, joint and velocity limits. The official
  `soma_to_g1.py` (Pink IK) only tracks body and hand targets and ignores the object apart from palm contact flags.

## Gaps in the baseline these map onto

1. The **reward has no object orientation**. `object_position` (std 0.2 m) is the only object term, and
   termination allows up to 1.5 rad of orientation error. Every leaderboard metric depends on orientation:
   ADD/AUC and MPPE use object points, SP-SR requires ≤0.5 rad, MP-SR ≤30°.
2. **Object contact is rewarded only at the palms** (force closure). Feet, legs and torso get nothing, yet the
   snack-box failure is a foot/leg pass.
3. **Retargeting ignores the object geometry.** G1 is about 1.3 m tall, much smaller than the human, so
   hand, foot and object relations get distorted.
4. A **1-world evaluation is noisy**: re-evaluating one checkpoint moved MPPE from 80 cm to 11 cm. Method
   comparisons need many worlds.

## Steps

| # | Step | Change | Output |
|---|---|---|---|
| S0 | Stable evaluation | Kit evaluator with N=32 worlds per checkpoint (same code, `evaluation.world_count`); report mean ± std | Baseline numbers for both sequences |
| S1 | RL: object point-cloud reward (ResMimic) | New `object_keypoints` term in the recon_body objective: 6 points at ±r along the object axes (r = object radius), Gaussian on the mean squared distance. Termination on the same keypoint distance instead of only position | Retrain both sequences, compare to S0 |
| S2 | RL: whole-body object contact reward (ResMimic) | Reward contact between the object and the links that the reference has near it (feet, shins, torso), not just the palms | Retrain snack box (and table) |
| S3 | Retargeting: OmniRetarget | Port `holosoma_retargeting`'s interaction-mesh solver to our SOMA input → G1 body joints. Keep the official parquet writer and the hand joints, and recompute end-effector frames by FK | New motion parquets; retrain with the best RL from S1/S2 |
| S4 | Combine, then seeds | Best retargeting + best RL; 2-3 seeds, choose by S0 evaluation | Final recipe for the 30-episode run |

Every training run: Grace A100, `slurm/train_a100.sbatch` (official recipe ≈ 3.5–4.5 h per sequence), and logs
in `metrics.jsonl`. Every comparison uses the S0 evaluator and also the 1-world kit score that the leaderboard uses.
Results are recorded in `docs/RESULTS.md`.
