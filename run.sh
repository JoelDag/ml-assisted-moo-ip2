#!/bin/bash

LOGDIR="runs"
mkdir -p "$LOGDIR"
JOBS=32
NO_PARALLEL="" #bc default is parallel

LOGFILE="run.log"
touch "$LOGFILE"

nohup python -m src.main \
  --logdir "$LOGDIR" \
  --jobs "$JOBS" \
  $NO_PARALLEL > "$LOGFILE" 2>&1 &


PID=$!
echo "Process started with PID: $PID"
echo "Logging to: $LOGFILE"
echo "To stop the process kill $PID"
tail -f "$LOGFILE"