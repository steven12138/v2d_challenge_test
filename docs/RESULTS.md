# Results: improving retargeting and RL

Plan and rationale: [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md). Test sequences: `round_table_flip_08` (RT) and
`snack_box_desk_through_06` (SB), Tier 1.

## Stage 1: retargeting (kinematic, no RL)

`retarget/eval_retarget.py`. Every motion is scored on the same G1 (sphere hands) + CoACD object scene.

| Metric | SB official (Pink IK) | SB OmniRetarget | RT official | RT OmniRetarget |
|---|---|---|---|---|
| Frames with object penetration > 1 cm | 24.6% | **0%** | 77.2% | (pending) |
| Object penetration mean / max | 0.65 / 4.8 cm | **0.04 / 0.14 cm** | 2.8 / 7.2 cm | |
| Frames with ground penetration > 1 cm | 7.3% | **0%** | 0.6% | |
| Left hand contact recall / distance | 99% / 0.6 cm | 100% / 1.4 cm | 94% / 0.5 cm | |
| Right hand contact recall / distance | 93% / 1.3 cm | 94% / 2.4 cm | 100% / 0.1 cm | |
| Foot sliding | 4.4 cm/s | 8.1 cm/s (worse) | 4.7 cm/s | |
| Joint-limit frames | 0 | 0 | 0 | |

Official contact distances look small because the hands sit inside the object. Keyframes (MuJoCo, 3 views):
`videos/keyframes/snack_box_official_vs_omni.png`. In the official SB motion, the forearms go into the box at
frames 193, 239, 720 and 774; with OmniRetarget the palms rest on the box sides. Video:
`videos/snack_box_desk_through_06/compare_retarget.mp4`.

## Stage 2: RL (FlashCHORD), 32-world kit evaluator

(pending: baseline vs object-keypoint reward, 3 seeds each; OmniRetarget motion + baseline RL)
