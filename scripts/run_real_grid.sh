#!/bin/bash
# Baseline (notebook-compatible) real-data grid: keep_first truncation, no padding mask.
# Each variant runs in its own process; a failure is logged and the next variant still runs.
cd "$(dirname "$0")/.."
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1
for spec in "retain withlabs" "retain nolabs" "transformer withlabs" "transformer nolabs"; do
  set -- $spec
  log="reports/experiments/logs/$1_$2.log"
  echo "[$(date '+%H:%M:%S')] start $1 $2" | tee -a reports/experiments/logs/grid.log
  .venv/bin/python -m aki.model.experiment --data "outputs/sequences_$2.parquet" --model "$1" \
      --device mps --experiment aki-real-baseline \
      --results-out "reports/experiments/real_$1_$2.json" > "$log" 2>&1
  echo "[$(date '+%H:%M:%S')] end $1 $2 exit=$?" | tee -a reports/experiments/logs/grid.log
done
