#!/usr/bin/env bash
# v9 FASE 2 (só iniciar com aprovação): chips -> cache -> CV C3(B) -> CV C2(A) -> C1 (v7 fora do fold) -> C4 (LightGBM) -> limiar global. PARA no fim.
cd "$(dirname "$0")"; PY=../.venv/Scripts/python.exe; LOG=logs/run_v9.log
step() { echo "=== $(date '+%F %T') | $*" | tee -a "$LOG"; "$PY" "$@" >> "$LOG" 2>&1; rc=$?; echo "=== $(date '+%F %T') | rc=$rc | $1" | tee -a "$LOG"; return $rc; }
step 02_chips_v9.py || exit 1
step 02c_build_cache.py || exit 1
step 11_oof_v7.py
step 08_spatial_cv.py --model cnn --folds 5 --epochs 60 --patience 15 --channels both --name unet_emb_B --meta chips_metadata_B.parquet
step 08_spatial_cv.py --model cnn --folds 5 --epochs 60 --patience 15 --channels both --name unet_emb_A --meta chips_metadata_A.parquet
step 12_lgbm_cv.py
step 10b_oof_threshold.py unet_emb_v7 unet_emb_A unet_emb_B lgbm_A
echo "=== V9_FASE2_FIM $(date '+%F %T') (aguardando decisão do usuário)" | tee -a "$LOG"
