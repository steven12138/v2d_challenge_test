# Results of v1

Official evaluator and packer, mean of 3 runs of `swa_last3` per episode; SP = mean error <= 10 cm and <= 0.5 rad, MP = mean error < 3 cm and < 30 deg.
A single official run is not bitwise reproducible, so a fresh run can differ from these means. `CHECKPOINTS.tsv` holds the sha256 of each policy, `checkpoints_v1.json` its path.

| Episode | Sequence | AUC | SP | MP | MPPE cm |
|---|---|---|---|---|---|
| 0 | 2026-06-04_14-16-23_tall_bar_stool_tip_up_04 | 0.968 | 1.00 | 1.00 | 0.92 |
| 1 | 2026-05-18_17-06-21_vacuum_left_lean_02 | 0.194 | 0.00 | 0.00 | 38.91 |
| 2 | 2026-06-02_10-04-42_blue_easter_basket_place_10 | 0.763 | 1.00 | 1.00 | 2.07 |
| 3 | 2026-04-27_11-50-10_contigo_coffee_mug_handle_table_05 | 0.919 | 1.00 | 1.00 | 1.37 |
| 4 | 2026-04-24_11-40-52_baseball_bat_rotate_05 | 0.182 | 0.00 | 0.00 | 75.62 |
| 5 | 2026-03-18_19-36-56_g1_box_rotate_01 | 0.910 | 1.00 | 1.00 | 1.33 |
| 6 | 2026-06-01_11-26-44_monitor_rotate_05 | 0.882 | 1.00 | 1.00 | 1.78 |
| 7 | 2026-05-29_15-20-43_iron_round_table_02 | 0.228 | 0.00 | 0.00 | 13.32 |
| 8 | 2026-06-05_15-08-08_skinny_wood_chair_push_10 | 0.953 | 1.00 | 1.00 | 1.17 |
| 9 | 2026-06-04_16-45-04_big_red_bowl_under_skinny_wood_chair_03 | 0.564 | 1.00 | 1.00 | 2.95 |
| 10 | 2026-05-01_11-36-40_white_desk_lower_to_ground_10 | 0.869 | 1.00 | 1.00 | 1.71 |
| 11 | 2026-04-24_13-56-58_light_blue_book_pick_place_09 | 0.718 | 1.00 | 1.00 | 2.64 |
| 12 | 2026-04-25_17-20-04_black_platform_turn_05 | 0.939 | 1.00 | 1.00 | 1.08 |
| 13 | 2026-06-09_10-19-09_snack_box_desk_through_06 | 0.776 | 1.00 | 1.00 | 2.17 |
| 14 | 2026-05-28_14-44-09_round_table_flip_08 | 0.956 | 1.00 | 1.00 | 0.97 |
| 15 | 2026-05-06_11-06-15_wet_floor_sign_pick_place_ground_08 | 0.934 | 1.00 | 1.00 | 1.18 |
| 16 | 2026-03-12_15-06-36_spray_bottle_left_desk_ground_02 | 0.834 | 1.00 | 1.00 | 1.77 |
| 17 | 2026-05-01_17-04-55_traffic_cone_ground_table_02 | 0.692 | 1.00 | 0.67 | 3.14 |
| 18 | 2026-05-06_10-00-19_blue_trash_can_drag_sideways_08 | 0.929 | 1.00 | 1.00 | 1.26 |
| 19 | 2026-05-19_10-19-48_cane_swing_05 | 0.673 | 1.00 | 1.00 | 2.77 |
| 20 | 2026-06-08_16-57-34_rice_cooker_clockwise_04 | 0.946 | 1.00 | 1.00 | 1.04 |
| 21 | 2026-04-29_11-53-57_tall_bar_stool_pull_08 | 0.899 | 1.00 | 1.00 | 1.50 |
| 22 | 2026-05-22_16-52-36_step_ladder_drag_03 | 0.949 | 1.00 | 1.00 | 1.10 |
| 23 | 2026-06-03_15-08-03_toy_stroller_back_forth_ground_04 | 0.951 | 1.00 | 1.00 | 1.09 |
| 24 | 2026-05-29_15-43-20_watering_can_pour_right_02 | 0.972 | 1.00 | 1.00 | 0.72 |
| 25 | 2026-03-20_11-59-57_cyan_water_bottle_rotate_01 | 0.126 | 0.00 | 0.00 | 119.41 |
| 26 | 2026-06-05_16-05-08_skateboard_rotate_ground_10 | 0.950 | 1.00 | 1.00 | 0.66 |
| 27 | 2026-05-01_11-15-05_woven_basket_ground_desk_10 | 0.729 | 0.67 | 0.00 | 9.71 |
| 28 | 2026-05-19_11-27-07_squeegee_sweep_away_from_you_03_ | 0.558 | 1.00 | 0.00 | 5.28 |
| 29 | 2026-06-05_18-27-40_rolling_luggage_left_foot_push_06 | 0.836 | 1.00 | 1.00 | 2.16 |

Mean over 30 episodes: AUC 0.760, SP 0.856, MP 0.789, MPPE 10.03 cm.

Episode 14 used keypoint std 0.05 m (`dev/pink_kp05`). Known failures: episodes 1, 4, 7, 25.
