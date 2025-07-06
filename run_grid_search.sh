#!/bin/bash

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGDIR="runs/run_$TIMESTAMP"
export RUN_OUTPUT_DIR="$LOGDIR"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/run_$TIMESTAMP.log"


JOBS=32
NO_PARALLEL="" #bc default is parallel



nohup setsid python -m src.main \
  --logdir "$LOGDIR" \
  --jobs "$JOBS" \
  --grid-search \
  --wand \
  --parallel > "$LOGFILE" 2>&1 &


PID=$!
disown "$PID"
echo "Process started with PID: $PID"
echo "Logging to: $LOGFILE"
echo "To stop the process kill $PID"
tail -f "$LOGFILE"