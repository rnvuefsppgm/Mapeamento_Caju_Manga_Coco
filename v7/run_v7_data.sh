#!/usr/bin/env bash
# v7 — fase A: compostos Sentinel 2025 (treino + 6 municípios). Log: logs/run_v7.log
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe; LOG=logs/run_v7.log
step() { echo "=== $(date '+%F %T') | $*" | tee -a "$LOG"; "$PY" "$@" >> "$LOG" 2>&1; rc=$?; echo "=== $(date '+%F %T') | rc=$rc | $1" | tee -a "$LOG"; return $rc; }
step 01_download_training_data.py --vector Pol_Caju.shp --role ground_truth
for i in 1 2 3; do step 04_download_prediction_data.py --vector municipios.shp --name-field NM_MUN && break; done   # 04 é retomável
echo "=== FASE_A_FIM $(date '+%F %T')" | tee -a "$LOG"
