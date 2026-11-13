#!/usr/bin/env bash
# Revision downloads (rev1). Re-runs the collector until every step is complete: exit code 2 means that the daily
# OpenAlex credit budget is exhausted on every key, in which case the script waits and tries again (finished batches
# are kept on disk and never requested twice). Any other failure stops the script after three attempts.
#   ./run_collect_rev1.sh [steps]        default steps: cutrefs,issn,authorworks
set -u
cd "$(dirname "$0")"
STEPS="${1:-cutrefs,issn,authorworks}"
WAIT="${WAIT_SECONDS:-1800}"
fails=0
while true; do
    .venv/bin/python collect_rev1.py --profile full --step "$STEPS" >> logs/collect_rev1_full.out 2>&1
    code=$?
    echo "$(date '+%F %T') collect_rev1 exit=$code steps=$STEPS" >> logs/collect_rev1.status
    if [ "$code" -eq 0 ]; then exit 0; fi
    if [ "$code" -eq 2 ]; then sleep "$WAIT"; continue; fi
    fails=$((fails + 1))
    if [ "$fails" -ge 3 ]; then exit "$code"; fi
    sleep 120
done
