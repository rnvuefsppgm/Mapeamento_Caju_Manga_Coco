#!/usr/bin/env bash
# v10b completo: MapBiomas (30, 31) -> chips (03) -> cache (02c, 02d) -> CV ajuste fino com trava (08, 10b) -> fusão (17)
# -> modelo final 100 épocas x 5 sementes (14) -> predição 25 (15) -> mosaicos (32) -> comparações (16, 33) -> mapa final opção c (34)
export GPU_AUG=1   # conversão, normalização e aumento de dados na GPU (src/ft.gpu_prep)
cd "$(dirname "$0")"; PY=../.venv/Scripts/python.exe; LOG=logs/run_v10b.log; mkdir -p logs
step() { echo "=== $(date '+%F %T') | $*" | tee -a "$LOG"; "$PY" "$@" >> "$LOG" 2>&1; rc=$?; echo "=== $(date '+%F %T') | rc=$rc | $1" | tee -a "$LOG"; return $rc; }
step 30_mapbiomas_recorte.py || exit 1
step 31_rotulos_mb.py || exit 1
step 03_chips.py || exit 1
step 02c_build_cache.py || exit 1
step 02d_cache_u8.py || exit 1
FT_INIT=../v10/models/ensemble_unet_conc step 08_spatial_cv.py --model cnn --folds 5 --epochs 60 --patience 15 --channels both --name unet_conc_b --meta chips_metadata_conc.parquet
step 10b_oof_threshold.py unet_emb_v7 unet_conc_b
IOU=$($PY -c "import json;print(json.load(open('reports/oof_unet_conc_b.json'))['iou'])")
echo "=== TRAVA: IoU CV v10b = $IOU (minimo 0.65)" | tee -a "$LOG"
$PY -c "import sys;sys.exit(0 if $IOU>=0.65 else 1)" || { echo "=== V10B_TRAVA_FALHOU" | tee -a "$LOG"; exit 2; }
step 17_fusao.py || exit 1
TH=$($PY -c "import json;print(json.load(open('reports/fusao.json'))['escolha']['limiar'])")
step 14_final_train.py 100 "$TH" || exit 1
step 15_predict_final.py || exit 1
step 32_mosaico.py
step 16_comparacao.py
step 33_unet_so.py
step 34_mapa_final_c.py
echo "=== V10B_FIM $(date '+%F %T')" | tee -a "$LOG"
