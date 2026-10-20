#!/usr/bin/env bash
# Train several transformer configs one after another. Logs -> results/logs/<config>.log
#   bash scripts/run_local_queue.sh configs/phobert.yaml configs/visobert.yaml
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
for cfg in "$@"; do
  name=$(basename "$cfg" .yaml)
  echo "=== $(date '+%F %T') start $name"
  python -u -m vigo train-transformer --config "$cfg" > "results/logs/$name.log" 2>&1
  echo "=== $(date '+%F %T') done  $name (exit $?)"
done
