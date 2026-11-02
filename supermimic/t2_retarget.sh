#!/usr/bin/env bash
# Retarget the filtered Tier 2 input (t2_filter.py) to G1 with the official Pink IK pipeline, on the CPU.
# usage: t2_retarget.sh <filtered tier dir> <motion root> <episode> [<episode> ...]
set -u
TIER=$(readlink -f "$1") OUT=$(readlink -f -m "$2")
shift 2
ROOT=$HOME/track2/retargeting
export PATH=$HOME/track2/.venv/bin:$PATH PYTHONPATH=$ROOT
mkdir -p "$OUT/whole_body/reconstructed_stage"
cd "$HOME/track2" || exit 1
for E in "$@"; do
  python "$ROOT/scripts/track2_to_soma.py" "$TIER" "$OUT/soma_inputs" --episodes "$E" > /dev/null || { echo "ep$E input failed"; continue; }
  SEQ=$(python3 -c "
import json
for l in open('$TIER/meta/episodes_metadata.jsonl'):
    d = json.loads(l)
    if d.get('episode_index') == $E:
        print(d.get('sequence_id') or d.get('source_sequence') or d.get('sequence'))
")
  python "$ROOT/scripts/retarget/soma_to_g1.py" "$OUT/soma_inputs/$SEQ" --save --motion-root "$OUT" > "$OUT/retarget_ep$E.log" 2>&1 \
    && cp "$OUT/soma_inputs/$SEQ/support.usda" "$OUT/whole_body/reconstructed_stage/${SEQ}_support.usda" \
    && echo "ep$E $SEQ retargeted" || echo "ep$E $SEQ failed, see $OUT/retarget_ep$E.log"
done
