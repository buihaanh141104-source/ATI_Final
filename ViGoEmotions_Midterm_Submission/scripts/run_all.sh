#!/usr/bin/env bash
# Reproduce every Midterm artefact from scratch (same seeds / split / configs).
#   bash scripts/run_all.sh                 # required models + extras + analysis
#   SKIP_EXTRA=1 bash scripts/run_all.sh    # only the three required models
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
mkdir -p results/logs

python -m vigo env                                   # hardware_and_versions.txt
python -m vigo eda                                   # WP1 tables + figures
python -m vigo split --make iterative                # WP2 split ids + audit
python -m vigo train-tfidf       --config configs/tfidf_lr.yaml | tee results/logs/tfidf_lr.log
python -m vigo train-rnn         --config configs/bilstm.yaml   | tee results/logs/bilstm.log
python -m vigo train-transformer --config configs/phobert.yaml  | tee results/logs/phobert.log

EXTRA=""
if [ -z "${SKIP_EXTRA:-}" ]; then
  python -m vigo train-transformer --config configs/visobert.yaml     | tee results/logs/visobert.log
  python -m vigo train-transformer --config configs/visobert14gb.yaml | tee results/logs/visobert14gb.log
  python -m vigo ensemble --runs phobert visobert visobert14gb --name ensemble_3
  python -m vigo ensemble-select > /dev/null        # validation-only composition table
  EXTRA="visobert visobert14gb ensemble_3"
fi
python -m vigo analyze --tfidf tfidf_lr --rnn bilstm --transformer phobert --extra $EXTRA | tee results/logs/analysis.log
