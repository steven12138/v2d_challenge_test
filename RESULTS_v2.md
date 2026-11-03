# Results of Tier 1 v2

Official evaluator and packer, mean of the official runs of each chosen policy (n runs); SP = mean error <= 10 cm and <= 0.5 rad, MP = mean error < 3 cm and < 30 deg.
A single official run is not bitwise reproducible, so a fresh run can differ from these means. `CHECKPOINTS_v2.tsv` holds the sha256 of each policy, `checkpoints_v2.json` its path.

| Episode | Sequence | n | AUC | SP | MP | MPPE cm | Policy |
|---|---|---|---|---|---|---|---|
| 0 | 2026-06-04_14-16-23_tall_bar_stool_tip_up_04 | 3 | 0.968 | 1.00 | 1.00 | 0.92 | `kp_ep0/swa_last3.safetensors` |
| 1 | 2026-05-18_17-06-21_vacuum_left_lean_02 | 3 | 0.194 | 0.00 | 0.00 | 38.91 | `kp_ep1/swa_last3.safetensors` |
| 2 | 2026-06-02_10-04-42_blue_easter_basket_place_10 | 3 | 0.763 | 1.00 | 1.00 | 2.07 | `kp_ep2/swa_last3.safetensors` |
| 3 | 2026-04-27_11-50-10_contigo_coffee_mug_handle_table_05 | 3 | 0.919 | 1.00 | 1.00 | 1.37 | `kp_ep3/swa_last3.safetensors` |
| 4 | 2026-04-24_11-40-52_baseball_bat_rotate_05 | 3 | 0.182 | 0.00 | 0.00 | 75.62 | `kp_ep4/swa_last3.safetensors` |
| 5 | 2026-03-18_19-36-56_g1_box_rotate_01 | 3 | 0.910 | 1.00 | 1.00 | 1.33 | `kp_ep5/swa_last3.safetensors` |
| 6 | 2026-06-01_11-26-44_monitor_rotate_05 | 3 | 0.882 | 1.00 | 1.00 | 1.78 | `kp_ep6/swa_last3.safetensors` |
| 7 | 2026-05-29_15-20-43_iron_round_table_02 | 6 | 0.801 | 1.00 | 1.00 | 2.39 | `flash_ep7_resume2/official/swa_300.safetensors` |
| 8 | 2026-06-05_15-08-08_skinny_wood_chair_push_10 | 3 | 0.953 | 1.00 | 1.00 | 1.17 | `kp_ep8/swa_last3.safetensors` |
| 9 | 2026-06-04_16-45-04_big_red_bowl_under_skinny_wood_chair_03 | 3 | 0.564 | 1.00 | 1.00 | 2.95 | `kp_ep9/swa_last3.safetensors` |
| 10 | 2026-05-01_11-36-40_white_desk_lower_to_ground_10 | 3 | 0.869 | 1.00 | 1.00 | 1.71 | `kp_ep10/swa_last3.safetensors` |
| 11 | 2026-04-24_13-56-58_light_blue_book_pick_place_09 | 3 | 0.718 | 1.00 | 1.00 | 2.64 | `kp_ep11/swa_last3.safetensors` |
| 12 | 2026-04-25_17-20-04_black_platform_turn_05 | 3 | 0.939 | 1.00 | 1.00 | 1.08 | `kp_ep12/swa_last3.safetensors` |
| 13 | 2026-06-09_10-19-09_snack_box_desk_through_06 | 3 | 0.776 | 1.00 | 1.00 | 2.17 | `kp_ep13/swa_last3.safetensors` |
| 14 | 2026-05-28_14-44-09_round_table_flip_08 | 3 | 0.956 | 1.00 | 1.00 | 0.97 | `kp_ep14/swa_last3.safetensors` |
| 15 | 2026-05-06_11-06-15_wet_floor_sign_pick_place_ground_08 | 3 | 0.934 | 1.00 | 1.00 | 1.18 | `kp_ep15/swa_last3.safetensors` |
| 16 | 2026-03-12_15-06-36_spray_bottle_left_desk_ground_02 | 3 | 0.834 | 1.00 | 1.00 | 1.77 | `kp_ep16/swa_last3.safetensors` |
| 17 | 2026-05-01_17-04-55_traffic_cone_ground_table_02 | 3 | 0.699 | 1.00 | 1.00 | 2.97 | `kp_ep17/w_320.safetensors` |
| 18 | 2026-05-06_10-00-19_blue_trash_can_drag_sideways_08 | 3 | 0.929 | 1.00 | 1.00 | 1.26 | `kp_ep18/swa_last3.safetensors` |
| 19 | 2026-05-19_10-19-48_cane_swing_05 | 1 | 0.709 | 1.00 | 1.00 | 2.53 | `grasp2_ep19_anchor/checkpoints/policy_final_26050560.safetensors` |
| 20 | 2026-06-08_16-57-34_rice_cooker_clockwise_04 | 3 | 0.946 | 1.00 | 1.00 | 1.04 | `kp_ep20/swa_last3.safetensors` |
| 21 | 2026-04-29_11-53-57_tall_bar_stool_pull_08 | 3 | 0.899 | 1.00 | 1.00 | 1.50 | `kp_ep21/swa_last3.safetensors` |
| 22 | 2026-05-22_16-52-36_step_ladder_drag_03 | 3 | 0.949 | 1.00 | 1.00 | 1.10 | `kp_ep22/swa_last3.safetensors` |
| 23 | 2026-06-03_15-08-03_toy_stroller_back_forth_ground_04 | 3 | 0.951 | 1.00 | 1.00 | 1.09 | `kp_ep23/swa_last3.safetensors` |
| 24 | 2026-05-29_15-43-20_watering_can_pour_right_02 | 3 | 0.972 | 1.00 | 1.00 | 0.72 | `kp_ep24/swa_last3.safetensors` |
| 25 | 2026-03-20_11-59-57_cyan_water_bottle_rotate_01 | 1 | 0.131 | 0.00 | 0.00 | 50.00 | `sonic_baseline/ep25/policy_sonic_only.safetensors` |
| 26 | 2026-06-05_16-05-08_skateboard_rotate_ground_10 | 3 | 0.950 | 1.00 | 1.00 | 0.66 | `kp_ep26/swa_last3.safetensors` |
| 27 | 2026-05-01_11-15-05_woven_basket_ground_desk_10 | 1 | 0.723 | 1.00 | 0.00 | 6.13 | `grasp2_ep27_rel/checkpoints/policy_final_13025280.safetensors` |
| 28 | 2026-05-19_11-27-07_squeegee_sweep_away_from_you_03_ | 3 | 0.558 | 1.00 | 0.00 | 5.28 | `kp_ep28/swa_last3.safetensors` |
| 29 | 2026-06-05_18-27-40_rolling_luggage_left_foot_push_06 | 3 | 0.836 | 1.00 | 1.00 | 2.16 | `kp_ep29/swa_last3.safetensors` |

Mean over 30 episodes: AUC 0.780, SP 0.900, MP 0.833, MPPE 7.22 cm (v1: 0.760, 0.856, 0.789, 10.03).
