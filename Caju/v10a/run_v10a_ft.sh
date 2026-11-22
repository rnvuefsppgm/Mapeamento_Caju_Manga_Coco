#!/usr/bin/env bash
# v10a-ft: CV ajuste fino (pesos v10 fold a fold) -> limiar -> trava -> fusão (OOF SM) -> final (3 sementes) -> predição 25 -> comparação
cd "$(dirname "$0")"; PY=../.venv/Scripts/python.exe; LOG=logs/run_v10a.log
step() { echo "=== $(date '+%F %T') | $*" | tee -a "$LOG"; "$PY" "$@" >> "$LOG" 2>&1; rc=$?; echo "=== $(date '+%F %T') | rc=$rc | $1" | tee -a "$LOG"; return $rc; }
FT_INIT=../v10/models/ensemble_unet_conc step 08_spatial_cv.py --model cnn --folds 5 --epochs 80 --patience 20 --channels both --name unet_conc_a --meta chips_metadata_conc.parquet
step 10b_oof_threshold.py unet_emb_v7 unet_conc_a
IOU=$($PY -c "import json;print(json.load(open('reports/oof_unet_conc_a.json'))['iou'])")
echo "=== TRAVA: IoU CV v10a-ft = $IOU (minimo 0.65)" | tee -a "$LOG"
$PY -c "import sys;sys.exit(0 if $IOU>=0.65 else 1)" || { echo "=== V10A_TRAVA_FALHOU" | tee -a "$LOG"; exit 2; }
step 17_fusao.py || exit 1
EP=$($PY -c "import json;print(json.load(open('reports/oof_unet_conc_a.json'))['epocas'] or 28)"); TH=$($PY -c "import json;print(json.load(open('reports/oof_unet_conc_a.json'))['limiar'])")
echo "=== modelo final: epocas=$EP limiar=$TH" | tee -a "$LOG"
step 14_final_train.py "$EP" "$TH" || exit 1
step 15_predict_final.py || exit 1
step 16_comparacao.py
echo "=== V10A_FIM $(date '+%F %T')" | tee -a "$LOG"
