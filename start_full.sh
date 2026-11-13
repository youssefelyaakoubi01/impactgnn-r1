#!/usr/bin/env bash
# Start the full pipeline (collect -> build_graph -> experiments -> summarize) detached from the terminal.
# Safe to call twice: it refuses to start a second run while one is in progress.
cd "$(dirname "$0")"
mkdir -p logs
if ! flock -n logs/run_full.lock true; then
    echo "Already running. Status: $(cat logs/run_full.status 2>/dev/null)"
    exit 0
fi
setsid nohup bash run_full.sh >> logs/run_full.log 2>&1 < /dev/null &
echo "Started (pid $!). Follow with: tail -f $(pwd)/logs/run_full.log"
