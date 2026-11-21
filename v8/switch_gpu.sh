#!/usr/bin/env bash
cd "$(dirname "$0")"; PY=../.venv/Scripts/python.exe; LOG=logs/predict.log
cnt() { $PY -c "import json;print(len(json.load(open('reports/area_caju_2025.json',encoding='utf-8'))))"; }
n0=$(cnt); until [ "$(cnt)" -gt "$n0" ]; do sleep 20; done
powershell -c "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*04_predict_emb.py*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
echo "=== $(date '+%F %T') | troca p/ GPU (04b) apos $(cnt) municipios" | tee -a $LOG
t0=$(date +%s); $PY 04b_predict_emb_gpu.py >> $LOG 2>&1; echo "=== $(date '+%F %T') | 04b rc=$? em $(( ($(date +%s)-t0)/60 )) min" | tee -a $LOG
$PY 05_compare_pam.py >> $LOG 2>&1
echo "=== PREDICT_FIM $(date '+%F %T')" | tee -a $LOG
