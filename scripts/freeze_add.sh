#!/usr/bin/env bash
# Add one episode's policy to checkpoints/ of this frozen version and record its hash.
# usage: scripts/freeze_add.sh <episode> <policy.safetensors> [<note>]
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
EP=$1 SRC=$2 NOTE=${3:-swa_last3}
mkdir -p "$HERE/checkpoints"
cp "$SRC" "$HERE/checkpoints/ep$EP.safetensors"
touch "$HERE/CHECKPOINTS.tsv"
grep -v -P "^$EP\t" "$HERE/CHECKPOINTS.tsv" > "$HERE/CHECKPOINTS.tsv.tmp" || true
printf '%s\t%s\t%s\n' "$EP" "$(sha256sum "$HERE/checkpoints/ep$EP.safetensors" | cut -c1-64)" "$NOTE" >> "$HERE/CHECKPOINTS.tsv.tmp"
sort -n "$HERE/CHECKPOINTS.tsv.tmp" > "$HERE/CHECKPOINTS.tsv"
rm "$HERE/CHECKPOINTS.tsv.tmp"
