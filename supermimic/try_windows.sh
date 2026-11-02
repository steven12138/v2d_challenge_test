#!/usr/bin/env bash
# Tier 1 only: other averaging windows of one finished Grace run, each with 64-world statistics and 3 official runs.
# usage: try_windows.sh <episode> <sequence> <grace subdir, e.g. sub_pink_kp> <gpu>    runs on H100, policies already in logs/kp_ep<E>/raw
set -u
E=$1 SEQ=$2 GPU=$3
ROOT=$HOME/track2
PY=$HOME/video_to_data/robotic_grounding/flash_chord/.venv/bin/python
D=$ROOT/logs/kp_ep$E
export PYTHONPATH=$ROOT
cd "$ROOT" || exit 1
for n in $(ls "$D/raw" | grep -oE "policy_[0-9]+" | sed 's/policy_//' | sort -n); do
  [ -f "$D/policy_$n.safetensors" ] || JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= "$PY" -m supermimic.kp_checkpoint "$D/raw/policy_$n.safetensors" "$D/policy_$n.safetensors"
done
mapfile -t P < <(ls -v "$D"/policy_*.safetensors)
N=${#P[@]}
try() {  # try <name> <policies...>
  local name=$1
  shift
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= "$PY" -m supermimic.swa_policy "$D/$name.safetensors" "$@" > /dev/null 2>&1
  CUDA_VISIBLE_DEVICES=$GPU "$PY" -u -m supermimic.diag_eval_stats x "$D/$name.safetensors" 2>/dev/null | grep -E "$name" | cut -c1-170 > "$D/$name.stats"
  CUDA_VISIBLE_DEVICES=$GPU bash "$ROOT/supermimic/eval_episode.sh" "$D/$name.safetensors" "$D/eval_$name" "$E" 3 > "$D/eval_$name.log" 2>&1
  echo "$(date '+%F %T') ep$E gpu$GPU WINDOW $name: $(grep -h 'locally computed' "$D/eval_$name.log" | sed 's/locally computed: //; s/AUC=//; s/  SP-SR=/ SP /; s/  MP-SR=/ MP /; s/  MPPE=/ MPPE /' | tr '\n' ';') | $(tail -n 1 "$D/$name.stats" | grep -oE 'AUC [0-9.]+|MPPE [0-9.]+' | tr '\n' ' ')" >> "$ROOT/logs/pipeline.log"
}
try w_last2 "${P[$((N - 2))]}" "${P[$((N - 1))]}"
try w_last5 "${P[@]:$((N - 5)):5}"
try w_prev3 "${P[$((N - 4))]}" "${P[$((N - 3))]}" "${P[$((N - 2))]}"
try w_early3 "${P[$((N - 5))]}" "${P[$((N - 4))]}" "${P[$((N - 3))]}"
