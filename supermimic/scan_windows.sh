#!/usr/bin/env bash
# Tier 1 only: 64-world statistics of the average of 3 consecutive policies for other end points of one finished run, since the last ones are not always the best
# (ep 7: 0.783 at 300M, 0.736 at 330M). Runs on H100, the policies are in logs/kp_ep<E>/raw. Results: logs/windows.tsv (episode, window end in M steps, AUC, MPPE, SP, MP).
# usage: scan_windows.sh <gpu> <episode> <window end in M steps> [...]
set -u
GPU=$1 EP=$2
shift 2
ROOT=$HOME/track2
PY=$HOME/video_to_data/robotic_grounding/flash_chord/.venv/bin/python
D=$ROOT/logs/kp_ep$EP
export PYTHONPATH=$ROOT
cd "$ROOT" || exit 1
for n in $(ls "$D/raw" | grep -oE "policy_[0-9]+" | sed 's/policy_//' | sort -n); do
  [ -f "$D/policy_$n.safetensors" ] || JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= "$PY" -m supermimic.kp_checkpoint "$D/raw/policy_$n.safetensors" "$D/policy_$n.safetensors"
done
mapfile -t P < <(ls -v "$D"/policy_*.safetensors)
for END in "$@"; do
  LAST=-1
  for I in "${!P[@]}"; do
    M=$(basename "${P[$I]}" .safetensors | sed 's/policy_//')
    [ $((M / 1000000)) -le "$END" ] && LAST=$I
  done
  if [ "$LAST" -lt 2 ]; then
    continue
  fi
  NAME=w_$END
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= "$PY" -m supermimic.swa_policy "$D/$NAME.safetensors" "${P[$((LAST - 2))]}" "${P[$((LAST - 1))]}" "${P[$LAST]}" > /dev/null 2>&1
  OUT=$(CUDA_VISIBLE_DEVICES=$GPU "$PY" -u -m supermimic.diag_eval_stats x "$D/$NAME.safetensors" 2>/dev/null | grep -E "$NAME")
  AUC=$(echo "$OUT" | grep -oE " AUC [0-9.]+" | tail -n 1 | cut -d' ' -f3)
  MPPE=$(echo "$OUT" | grep -oE "MPPE [0-9.]+" | tail -n 1 | cut -d' ' -f2)
  SP=$(echo "$OUT" | grep -oE "SP [0-9.]+" | head -n 1 | cut -d' ' -f2)
  MP=$(echo "$OUT" | grep -oE "MP [0-9.]+" | head -n 1 | cut -d' ' -f2)
  flock "$ROOT/logs/windows.tsv.lock" sh -c "printf '%s\t%s\t%s\t%s\t%s\t%s\n' '$EP' '$END' '$AUC' '$MPPE' '$SP' '$MP' >> '$ROOT/logs/windows.tsv'"
done
