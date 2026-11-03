# Track 2 Tier 1 recipe, version 2

v1 (`README.md`, `RESULTS.md`) with the policy of 5 episodes replaced by a better one from follow-up runs on the same Tier 1 motion. Each episode keeps the policy with the best mean of its official runs (AUC + SP + MP - MPPE / 100, the ranking of `scripts/scan_official.py`, Tier 2 directories excluded). The other 25 episodes are the v1 policies, byte for byte.

## Replaced episodes

| Episode | Policy | How it was made |
|---|---|---|
| 7 | `flash_ep7_resume2/official/swa_300` | FlashSAC resumed from the 270M state of the v1 run with the NaN guard (`supermimic/train_guard.py`), then the average of the 3 policies ending at 300M (`supermimic/swa_policy.py`; window choice by `supermimic/scan_windows.sh`) |
| 17 | `kp_ep17/w_320` | the v1 run, average of the 3 policies ending at 320M instead of 350M (`supermimic/try_windows.sh`) |
| 19 | `grasp2_ep19_anchor` final | SuperMimic refinement with a trust region around `kp_ep19/swa_last3` (`supermimic/run_anchor.sh`: official and keypoint rewards, official termination, actor lr / 3, `w_anchor=20`) |
| 25 | `sonic_baseline/ep25/policy_sonic_only` | the zero-residual policy, i.e. the official SONIC controller alone (`supermimic/sonic_baseline.py`); every trained policy drops the bottle, and doing nothing loses it later |
| 27 | `grasp2_ep27_rel` final | the same trust-region refinement of `kp_ep27/swa_last3`, plus release terms: fingers open 25 frames before the labelled release (`grip_release_lead=25`) and 6x reset weight around the last release (`release_boost=6`) |

Every policy goes through `supermimic/kp_checkpoint.py`, so the unmodified evaluator rebuilds the official environment.

## Build

`CHECKPOINTS=checkpoints_v2.json scripts/build_submission.sh <name> <commit url> <gpu> [gpu ...]` gives each policy one fresh official run and packs the Tier 1 CSV.

## Files

| Path | What |
|---|---|
| `checkpoints_v2.json` | episode to policy path on the training machine |
| `CHECKPOINTS_v2.tsv` | episode, sha256, source of each policy |
| `RESULTS_v2.md` | mean official scores of the chosen policies |
