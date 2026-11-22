#!/usr/bin/env bash
cd "$(dirname "$0")"; PY=../.venv/Scripts/python.exe; LOG=logs/run_v10.log
step() { echo "=== $(date '+%F %T') | $*" | tee -a "$LOG"; "$PY" "$@" >> "$LOG" 2>&1; rc=$?; echo "=== $(date '+%F %T') | rc=$rc | $1" | tee -a "$LOG"; return $rc; }
step 14_final_train.py 28 0.55 || exit 1
step 15_predict_final.py || exit 1
step 16_comparacao.py
echo "=== V10_FIM $(date '+%F %T')" | tee -a "$LOG"
