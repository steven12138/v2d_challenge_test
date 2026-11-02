# Results of v1

64 worlds, official AUC and MPPE functions; SP = mean error <= 10 cm and <= 0.5 rad, MP = mean error < 3 cm and < 30 deg.
Two 64-world runs of one policy can differ by about 0.04 AUC and 10 % MP.

| Episode | Training output | AUC | SP | MP | MPPE cm |
|---|---|---|---|---|---|
| 14 round_table_flip_08 | `dev/pink_kp05` (keypoint std 0.05 m) | 0.955 | 1.00 | 1.00 | 1.00 |
| 23 toy_stroller_back_forth_ground_04 | `dev/pink_kp` | 0.953 | 1.00 | 1.00 | 1.06 |
| 10 white_desk_lower_to_ground_10 | `dev/pink_kp` | 0.859 | 1.00 | 1.00 | 1.83 |
| 13 snack_box_desk_through_06 | `sub_pink_kp` | 0.718-0.760 | 0.92-0.98 | 0.83-0.94 | 2.67-4.06 |
| 19 cane_swing_05 | `dev/pink_kp` | 0.672-0.674 | 1.00 | 0.91-0.95 | 2.83-2.88 |

Official evaluator, 3 runs each: episode 13 AUC 0.840 / 0.741 / 0.743, episode 19 AUC 0.677 / 0.665 / 0.681, SP and MP 1 on all of them.

Known failure: episode 25 cyan_water_bottle_rotate_01, AUC 0.12, all worlds lose the bottle by frame 170 (the open hand knocks it away).
The other 24 episodes are still training at the time of this freeze.
