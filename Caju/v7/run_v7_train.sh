#!/usr/bin/env bash
# v7 — 2025, SÓ ÁREA DE TREINO (3 clusters): embeddings -> chips -> pseudo -> CV x3 (GPU) -> comparação -> mapa.
# Uso: ./run_v7_train.sh rnv-geodatin | Log: logs/run_v7.log
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe; LOG=logs/run_v7.log; PROJ="$1"
step() { echo "=== $(date '+%F %T') | $*" | tee -a "$LOG"; "$PY" "$@" >> "$LOG" 2>&1; rc=$?; echo "=== $(date '+%F %T') | rc=$rc | $1" | tee -a "$LOG"; return $rc; }
until [ -f metadata/training_provenance_2025.parquet ]; do sleep 60; done   # espera o 01 (2025) terminar
step 03a_download_embeddings.py --project "$PROJ" || exit 1
step 02_create_training_chips.py --vector Pol_Caju.shp || exit 1
step 02b_pseudo_label_chips.py --vector Pol_Caju.shp
for M in "emb unet_emb" "sentinel unet_sentinel" "both unet_sentinel_emb"; do
  set -- $M; step 08_spatial_cv.py --model cnn --folds 5 --epochs 60 --patience 15 --channels "$1" --name "$2"
done
step 10_compare_models.py
# mapa da área de treino: provenance de predição = COGs dos clusters 2025
"$PY" -c "import pandas as pd; d=pd.read_parquet('metadata/training_provenance_2025.parquet'); d.assign(municipality=d.cluster_id)[['municipality','features_cog']].to_parquet('metadata/prediction_provenance.parquet', index=False)"
BEST=$(cat reports/best_model.txt 2>/dev/null)
[ -n "$BEST" ] && step 05_predict.py --ensemble "$BEST"
echo "=== FASE_B_FIM $(date '+%F %T')" | tee -a "$LOG"
