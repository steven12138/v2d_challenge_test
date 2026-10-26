# Results: improving retargeting and RL

Plan and rationale: [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md). Test sequences: `round_table_flip_08` (RT,
ground only) and `snack_box_desk_through_06` (SB, desk support), Tier 1.

## Stage 1: retargeting (kinematic, no RL)

`retarget/eval_retarget.py` scores every motion on the trained robot (official G1 + Dex3, all 43 joints) plus the
CoACD-decomposed object, using MuJoCo signed distances. "Without fingers" excludes the Dex3 finger links, which the
retargeters leave at 0 and which the RL finger residual moves.

| Metric | SB Pink IK (official) | SB OmniRetarget | RT Pink IK (official) | RT OmniRetarget |
|---|---|---|---|---|
| Frames with object penetration > 1 cm (no fingers) | 27.2% | **1.6%** | 77.5% | **2.9%** |
| Object penetration mean, no fingers | 0.75 cm | **0.12 cm** | 2.59 cm | **0.17 cm** |
| Frames with object penetration > 1 cm (all links) | 38.5% | 30.5% | 84.5% | 47.1% |
| Frames with ground penetration > 1 cm | 4.7% | **0%** | 0.2% | **0%** |
| Left / right hand contact recall | 100 / 100% | 100 / 100% | 100 / 100% | 94 / **66%** |
| Foot sliding | 4.3 cm/s | 8.8 cm/s | 5.2 cm/s | 5.8 cm/s |
| Foot height, median | 0.3 cm | 0.8-1.2 cm | 0.3 cm | 1.4-1.8 cm |

OmniRetarget here (`retarget/omniretarget/omniretarget.py`) does not use Pink IK at all. It solves
interaction-mesh SQP with the official object trajectory kept exact and the demonstration scaled in the object
frame. Ground points in the mesh stop the robot from floating: without them it hovered 6 cm high on RT. An
infeasible SQP step falls back to dropping foot sticking, then to holding the pose. Remaining issues: the
RT right-hand contact recall, and foot sliding on SB.

Keyframes (3 views): `videos/keyframes/snack_box_official_vs_omni.png`. In the official SB motion the forearms sink
into the box at frames 193, 239, 720 and 774; with OmniRetarget the palms rest on the box sides.

## Stage 2: RL (FlashCHORD)

32-world kit evaluator (`slurm/eval_worlds.sbatch`), official recipe, 350M steps. 1-world scores are very noisy:
RT A40 scored MPPE 3.4 cm on one world but averages 17.5 cm over 32.

| Run | AUC | SP-SR | MP-SR | MPPE | pos err | ori err |
|---|---|---|---|---|---|---|
| RT baseline, A40 | 0.358 | 0.69 | 0 | 17.5 cm | 17.4 cm | 25.7° |
| RT baseline, A100, replay 5M | 0.324 | 0.59 | 0 | 13.7 cm | 13.6 cm | 33.1° |
| SB baseline, A100 | 0.303 | 0.31 | 0 | 17.7 cm | 17.5 cm | 24.3° |

Object-keypoint reward (`g1_recon_body_kp_flash_sac`), training metrics at 60-80M steps on SB (VOC 0.75-0.5):

| Run | object pos err | object ori err | reaches end | object terminations |
|---|---|---|---|---|
| baseline s1 / s2 | 3.9 / 3.9 cm | 0.061 / 0.059 rad | 71 / 73% | 16 / 16% |
| keypoints s42 / s1 / s2 | **2.3 / 2.1 / 2.2 cm** | **0.051 / 0.044 / 0.045 rad** | 71 / 76 / 76% | 15 / 10 / 11% |

## Tier 1 submission run

All 30 sequences: OmniRetarget motion + object-keypoint recipe, seed 42, Grace A100 (`scripts/launch_tier.sh`,
work/tier1/sub_omni_kp). Code: github.com/steven12138/v2d_challenge_test.
