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

## Stage 2: RL (FlashCHORD), 32-world kit evaluator

`slurm/eval_worlds.sbatch` runs the official evaluator over 32 worlds (1-world scores are very noisy: RT A40 scored
MPPE 3.4 cm on one world but averages 17.5 cm). All runs: 350M steps, seed 42.

| Run | retarget | reward | AUC | SP-SR | MP-SR | MPPE |
|---|---|---|---|---|---|---|
| SB baseline (A100) | Pink IK | official | 0.303 | 0.31 | 0.00 | 17.7 cm |
| SB omni v1 | OmniRetarget refining Pink | official | 0.430 | 0.69 | 0.00 | 9.9 cm |
| **SB kp** | Pink IK | **object keypoints** | **0.722** | **0.91** | **0.78** | **4.2 cm** |
| RT baseline (A40 / A100) | Pink IK | official | 0.358 / 0.342 | 0.69 / 0.97 | 0 / 0 | 17.5 / 7.2 cm |
| **RT kp** | Pink IK | **object keypoints** | **0.788** | 0.91 | **0.84** | 10.0 cm |

Conclusions:
1. The point-cloud object reward (ResMimic) is the largest gain: AUC more than doubles, and MP-SR goes from 0 to
   ~0.8 on both sequences.
2. OmniRetarget refining the Pink-IK solution also helps with the official reward (SB SP-SR 0.31 -> 0.69).
3. Standalone OmniRetarget (no Pink IK) trains badly. Its legs deviate ~16 deg RMS from a human-like solution and
   SONIC cannot track them: pelvis error 0.15 m vs 0.06 m, and ~60% of episodes end on palm deviation. Smoothing
   the solution or fixing the ground points in the world does not change this.
   (training metrics: `evaluate/compare_runs.py`, `series.py`)
4. Not yet tested: OmniRetarget refine + object keypoints (`omniretarget.py ... <official parquet>`), which
   combines 1 and 2.

Early training signal (`metrics.jsonl`), SB at 60-80M steps: the keypoint reward halves the object position error
over 3 seeds (2.1-2.3 cm vs 3.9 cm).

## Tier 1 batches

| Batch | Status |
|---|---|
| `sub_omni_kp` (standalone OmniRetarget + kp) | cancelled at 210-330M steps (1/30 final): poor training metrics |
| `sub_omni3_kp` (+ 3 Hz smoothing) | cancelled at 150-210M steps |
| `sub_pink_kp` (Pink + kp) | only the 2 test sequences (above) |
| A/B launcher `scripts/launch_tier1_ab.sh` | ready, not started: A = Pink + kp, B = OmniRetarget refine + kp |
