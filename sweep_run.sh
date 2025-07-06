#!/bin/bash

export WANDB_MODE=online
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGDIR="runs/run_$TIMESTAMP"
mkdir -p "$LOGDIR"
export RUN_OUTPUT_DIR="$LOGDIR"

SWEEP_ID="joeldag-paderborn-university/ml4moo_topic2/14bwiiyc"
AGENTS=32
CONDA_ENV_NAME="mlamoo"

echo "Launching $AGENTS W&B agents in conda env '$CONDA_ENV_NAME'"

for i in $(seq 1 $AGENTS); do
  LOGFILE="$LOGDIR/agent_${i}.log"
  echo "Starting agent $i logging to $LOGFILE"

  nohup conda run -n $CONDA_ENV_NAME wandb agent $SWEEP_ID > "$LOGFILE" 2>&1 &

done

echo "Agents launched. Logs in $LOGDIR"
