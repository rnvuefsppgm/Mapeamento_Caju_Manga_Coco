#!/usr/bin/env bash
# v1 — depois do treino final: predição 25 municípios -> mosaico -> comparação -> vetores/KMZ -> retoma o COCO (pedido do usuário).
# Predição com GPU_FRAC=0,9 (o teto de 45% causou falta de memória; o coco está parado). Retomável. Log: logs/run_treino_v1.log
cd "$(dirname "$0")"; PY=D:/Modelo_Caju/.venv/Scripts/python.exe; LOG=logs/run_treino_v1.log
export PYTHONIOENCODING=utf-8 GPU_FRAC=0.9
step() { echo "=== $(date '+%F %T') | $*" >> "$LOG"; "$PY" "$@" >> "$LOG" 2>&1; rc=$?; echo "=== $(date '+%F %T') | rc=$rc | $1" >> "$LOG"; return $rc; }
TH=$("$PY" -c "import json;print(json.load(open('reports/oof_v1_A.json'))['limiar'])")
step 16_predicao.py "$TH" || { echo "=== V1_POS_ERRO predicao" >> "$LOG"; exit 1; }
step 17_mosaico.py
step 18_comparacao.py
step 19_vetores_kmz.py
echo "=== V1_FIM $(date '+%F %T')" >> "$LOG"
# ---- coco (só se não houver um 30_treino.py em python.exe rodando) ----
N=logs/coco_retomada.log
if powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? { \$_.CommandLine -match '30_treino' }) { exit 0 } else { exit 1 }"; then
  echo "$(date '+%F %T') coco já está rodando (outra sessão): não disparado" >> "$N"; exit 0
fi
echo "$(date '+%F %T') retomando coco: 30_treino.py cv 60 15" >> "$N"
cd D:/Backup_2026/Geral_desktop/Mapa_COCO/scripts || exit 1
unset GPU_FRAC
D:/Modelo_Caju/.venv/Scripts/python.exe -W ignore 30_treino.py cv 60 15 >> D:/Modelo_Coco/v1/logs/30_cv_retomada.log 2>&1
echo "$(date '+%F %T') coco CV terminou rc=$?" >> "D:/Backup_2026/Geral_desktop/MAPA_MANGA/$N"
