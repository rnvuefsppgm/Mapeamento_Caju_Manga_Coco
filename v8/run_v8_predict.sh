#!/usr/bin/env bash
# v8 — espera exportações + 5 folds do unet_emb (v7) -> VRT -> mapas 2025 dos 25 municípios -> comparação PAM.
cd "$(dirname "$0")"; PY=../.venv/Scripts/python.exe; LOG=logs/predict.log
until grep -q EXPORT_FIM logs/export.log 2>/dev/null; do sleep 120; done
until [ -f ../v7/models/ensemble_unet_emb/fold5.pt ]; do sleep 120; done
echo "=== $(date '+%F %T') | VRT" | tee -a $LOG; $PY 03_mosaic_vrt.py >> $LOG 2>&1
echo "=== $(date '+%F %T') | PREDICT" | tee -a $LOG; $PY 04_predict_emb.py >> $LOG 2>&1
echo "=== $(date '+%F %T') | PAM" | tee -a $LOG; $PY 05_compare_pam.py >> $LOG 2>&1
echo "=== PREDICT_FIM $(date '+%F %T')" | tee -a $LOG
