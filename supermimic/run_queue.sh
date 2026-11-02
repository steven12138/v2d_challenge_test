#!/usr/bin/env bash
# A GPU worker: takes episodes one at a time from a list shared with the other workers until it stays empty.
# Lines: "<episode> <sequence_id>" runs run_pipeline.sh, "<episode> <sequence_id> <policy> [run tag]" runs run_refine.sh on that policy.
# usage: run_queue.sh <gpu> <list> [wait_file wait_text]   (starts once wait_text appears in wait_file)
#        env IDLE_MIN=<minutes> keeps waiting for new lines that long before leaving (default 0)
set -u
GPU=$1 Q=$2
if [ $# -ge 4 ]; then
  until grep -q "$4" "$3" 2>/dev/null; do sleep 30; done
fi
IDLE=0
while true; do
  LINE=$(flock "$Q.lock" sh -c "head -1 '$Q'; sed -i 1d '$Q'")
  if [ -z "$LINE" ]; then
    [ "$IDLE" -ge "${IDLE_MIN:-0}" ] && break
    IDLE=$((IDLE + 1))
    sleep 60
    continue
  fi
  IDLE=0
  set -- $LINE
  if [ $# -ge 3 ]; then
    bash "$HOME/track2/supermimic/run_refine.sh" "$1" "$2" "$GPU" "$3" "${4:-_fw}"
  else
    bash "$HOME/track2/supermimic/run_pipeline.sh" "$1" "$2" "$GPU"
  fi
done
echo "$(date '+%F %T') gpu$GPU queue empty" >> "$HOME/track2/logs/pipeline.log"
