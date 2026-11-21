#!/usr/bin/env bash
# v10 completo: fontes(01, já rodando) -> rótulos -> amostras -> cache -> C1 fora do fold -> CV -> limiar global -> modelo final -> predição 25 -> comparação
cd "$(dirname "$0")"; PY=../.venv/Scripts/python.exe; LOG=logs/run_v10.log
step() { echo "=== $(date '+%F %T') | $*" | tee -a "$LOG"; "$PY" "$@" >> "$LOG" 2>&1; rc=$?; echo "=== $(date '+%F %T') | rc=$rc | $1" | tee -a "$LOG"; return $rc; }
until grep -q FONTES_FIM logs/01_fontes.log 2>/dev/null; do sleep 60; done
grep -q "FONTES_FIM rc=0" logs/01_fontes.log || { echo "=== ERRO nas fontes" | tee -a "$LOG"; exit 1; }
step 02_rotulos.py || exit 1
step 03_chips.py || exit 1
step 02c_build_cache.py || exit 1
step 11_oof_v7.py
step 08_spatial_cv.py --model cnn --folds 5 --epochs 60 --patience 15 --channels both --name unet_conc --meta chips_metadata_conc.parquet
step 10b_oof_threshold.py unet_emb_v7 unet_conc
EP=$($PY -c "import json;print(json.load(open('reports/oof_unet_conc.json'))['epocas'] or 40)"); TH=$($PY -c "import json;print(json.load(open('reports/oof_unet_conc.json'))['limiar'])")
echo "=== modelo final: epocas=$EP limiar=$TH" | tee -a "$LOG"
step 14_final_train.py "$EP" "$TH" || exit 1
step 15_predict_final.py || exit 1
step 16_comparacao.py
echo "=== V10_FIM $(date '+%F %T')" | tee -a "$LOG"
