#!/usr/bin/env bash
# Exporta embeddings 2025 (DatensEE, sem tarefas EE) para as regiões de regions_<lista>.txt. Retomável.
# uso: bash scripts/run_export.sh <lista> [regiao_unica]   (adaptado de D:/Modelo_Caju/v8/run_v8_export.sh)
# Caminhos RELATIVOS (cd na versão): o pipeline Java lê "D:/..." como esquema de URL e falha.
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
L=${1:-amostragem_inicial}; SO=$2; DS=../../Modelo_Caju/.venv_v8/Scripts/datensee; PROJ=rnv-geodatin
cd D:/Modelo_Coco/v1 || exit 1
mkdir -p logs; LOG=logs/export.log
while read -r R CRS || [ -n "$R" ]; do
  R=$(echo "$R" | tr -d '\r'); CRS=$(echo "$CRS" | tr -d '\r'); [ -z "$R" ] && continue
  [ -n "$SO" ] && [ "$R" != "$SO" ] && continue
  OUT=data/embeddings/2025/$R; mkdir -p "$OUT"
  if ls "$OUT"/*.tif >/dev/null 2>&1 && [ ! -s "$OUT/_failures.json" ]; then echo "EXPORT pulado $R" >> $LOG; continue; fi
  t0=$(date +%s)
  $DS export expressions/embeddings_2025.json data/aoi/regions/$R.geojson -p $PROJ -o "$OUT" \
     --scale 10 --crs $CRS --tile-size 512 --band-count 64 --data-type uint8 --nodata 0 -y >> $LOG 2>&1
  for i in 1 2 3; do [ -s "$OUT/_failures.json" ] || break; $DS retry -o "$OUT" >> $LOG 2>&1; done
  n=$(ls "$OUT"/*.tif 2>/dev/null | wc -l)
  echo "EXPORT $R $CRS | tiles=$n | $(( $(date +%s)-t0 ))s | $(du -sh "$OUT" | cut -f1)" | tee -a $LOG
done < data/aoi/regions_$L.txt
echo "EXPORT_FIM $L $(date '+%F %T')" | tee -a $LOG
