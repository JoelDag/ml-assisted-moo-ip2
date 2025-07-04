#!/bin/bash

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGDIR="runs/run_$TIMESTAMP"
LOGFILE="$LOGDIR/run_$TIMESTAMP.log"
mkdir -p "$LOGDIR"

JOBS=32
NO_PARALLEL="" #bc default is parallel



nohup python -m src.main \
  --logdir "$LOGDIR" \
  --jobs "$JOBS" \
  $NO_PARALLEL > "$LOGFILE" 2>&1 &


PID=$!
echo "Process started with PID: $PID"
echo "Logging to: $LOGFILE"
echo "To stop the process kill $PID"
tail -f "$LOGFILE"