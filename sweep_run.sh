#!/bin/bash

export WANDB_MODE=online
export WANDB_PROJECT=mlamoo_random_search
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGDIR="runs/NSGA_2_paper_setup_run_$TIMESTAMP"
mkdir -p "$LOGDIR"
export RUN_OUTPUT_DIR="$LOGDIR"

export WANDB_PROJECT=mlamoo_run_paper_params_NSAG2
SWEEP_ID="joeldag-paderborn-university/mlamoo_run_paper_params_NSAG2/sob8aod9"
AGENTS=22
CONDA_ENV_NAME="mlamoo"

echo "Launching $AGENTS W&B agents in conda env '$CONDA_ENV_NAME'"

for i in $(seq 1 $AGENTS); do
  LOGFILE="$LOGDIR/agent_${i}.log"
  echo "Starting agent $i logging to $LOGFILE"

  nohup conda run -n $CONDA_ENV_NAME --no-capture-output wandb agent $SWEEP_ID > "$LOGFILE" 2>&1 &

done

echo "Agents launched. Logs in $LOGDIR"
