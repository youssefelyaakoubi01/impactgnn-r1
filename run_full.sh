#!/usr/bin/env bash
# Full run, detached from the terminal: collection (retried after the OpenAlex daily reset),
# graph build, all experiments, summary. Launch with:
#   setsid nohup bash run_full.sh > logs/run_full.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
STATUS=logs/run_full.status
say() { echo "$(date '+%F %T') $*"; echo "$(date '+%F %T') $*" > "$STATUS"; }

# only one full run at a time
exec 9> logs/run_full.lock
flock -n 9 || { echo "run_full.sh is already running (see logs/run_full.status)"; exit 0; }

# wait for a collection already running in another shell
while pgrep -f "collect.py --profile full" > /dev/null; do sleep 30; done

say "collect: start"
for attempt in $(seq 1 40); do
    $PY collect.py --profile full --step seeds,topics,sample,authors,refs,nodes,venues
    code=$?
    if [ $code -eq 0 ]; then break; fi
    say "collect: attempt $attempt exited with $code (budget exhausted or network); retry in 20 min"
    sleep 1200
done
[ $code -eq 0 ] || { say "collect: FAILED after retries"; exit 1; }

say "build_graph: start"
$PY build_graph.py --profile full || { say "build_graph: FAILED"; exit 1; }
say "experiments: start"
$PY experiments.py --profile full --exp all --jobs 3 || { say "experiments: FAILED"; exit 1; }
$PY summarize.py --profile full || { say "summarize: FAILED"; exit 1; }
say "DONE: results/summary.md written"
