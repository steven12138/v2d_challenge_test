#!/usr/bin/env bash
# One trust-region refinement test with its EMA statistics in logs/pipeline.log.
# usage: test_anchor.sh <tag> <episode> <sequence_id> <gpu> <policy.safetensors> [extra overrides...]
set -u
RTAG=$1 EP=$2 SEQ=$3 GPU=$4 POLICY=$5
shift 5
D=$HOME/track2/logs/grasp2_ep${EP}${RTAG}
mkdir -p "$D"
nohup setsid bash "$HOME/track2/supermimic/watch_ema.sh" "$GPU" "$EP" "$D" > /dev/null 2>&1 < /dev/null &
TAG=$RTAG bash "$HOME/track2/supermimic/run_anchor.sh" "$EP" "$SEQ" "$GPU" "$POLICY" "$@"
