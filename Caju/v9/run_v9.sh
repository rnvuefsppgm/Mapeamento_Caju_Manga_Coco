#!/usr/bin/env bash
# v9 — só embeddings 2025: chips (campo + pseudo A/B) -> cache -> CV B (com pseudo) -> CV A (sem pseudo do modelo) -> limiar global. PARA aqui (aguarda usuário).
cd "$(dirname "$0")"; PY=../.venv/Scripts/python.exe; LOG=logs/run_v9.log
step() { echo "=== $(date '+%F %T') | $*" | tee -a "$LOG"; "$PY" "$@" >> "$LOG" 2>&1; rc=$?; echo "=== $(date '+%F %T') | rc=$rc | $1" | tee -a "$LOG"; return $rc; }
until grep -q PREDICT_FIM ../v8/logs/predict.log 2>/dev/null && grep -q MB_API_FIM logs/mapbiomas_api.log 2>/dev/null; do sleep 120; done
step 02_chips_v9.py || exit 1
step 02c_build_cache.py || exit 1
step 08_spatial_cv.py --model cnn --folds 5 --epochs 60 --patience 15 --channels both --name unet_emb_B --meta chips_metadata_B.parquet
step 08_spatial_cv.py --model cnn --folds 5 --epochs 60 --patience 15 --channels both --name unet_emb_A --meta chips_metadata_A.parquet
step 10b_oof_threshold.py unet_emb_B unet_emb_A
echo "=== V9_TREINO_FIM $(date '+%F %T') (aguardando decisão do usuário)" | tee -a "$LOG"
