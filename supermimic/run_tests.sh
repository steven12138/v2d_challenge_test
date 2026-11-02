#!/usr/bin/env bash
# The test GPU's worker: runs the commands of a list one at a time, waiting for new lines (the list can change while it runs).
# usage: run_tests.sh <list> [wait_file wait_text]    each line is a shell command that takes the GPU for itself
set -u
Q=$1
if [ $# -ge 3 ]; then
  until grep -q "$3" "$2" 2>/dev/null; do sleep 30; done
fi
touch "$Q"
while true; do
  LINE=$(flock "$Q.lock" sh -c "head -1 '$Q'; sed -i 1d '$Q'")
  if [ -z "$LINE" ]; then
    sleep 60
    continue
  fi
  echo "$(date '+%F %T') test start: $LINE" >> "$HOME/track2/logs/tests.log"
  bash -c "$LINE" >> "$HOME/track2/logs/tests.log" 2>&1
  echo "$(date '+%F %T') test end rc=$?: $LINE" >> "$HOME/track2/logs/tests.log"
done
