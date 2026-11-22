#!/usr/bin/env bash
# 04 — Exporta os embeddings AlphaEarth 2025 (DatensEE, High Volume API, sem tarefas EE) por região. Retomável.
# Igual ao caju v8 (export_one.sh + run_v8_export.sh): tiles 512 px, EPSG:32724, 10 m, uint8 (NoData 0), retry 3x.
# uso: ./04_export_embeddings.sh            (todas as regiões de dados/aoi/regioes.txt)
#      ./04_export_embeddings.sh <regiao>   (uma só, p/ teste)
cd "$(dirname "$0")"; DS=D:/Modelo_Caju/.venv_v8/Scripts/datensee; LOG=logs/04_export.log
EX=dados/aoi/expressions/embeddings_2025.json
um() {
  R=$1; OUT=dados/embeddings/$R; mkdir -p "$OUT"
  if ls "$OUT"/*.tif >/dev/null 2>&1 && [ ! -s "$OUT/_failures.json" ] && [ -f "$OUT/_export_meta.json" ]; then
    echo "EXPORT pulado $R (já existe)" >> "$LOG"; return; fi
  t0=$(date +%s)
  "$DS" export "$EX" dados/aoi/regioes/$R.geojson -p rnv-geodatin -o "$OUT" \
     --scale 10 --crs EPSG:32724 --tile-size 512 --band-count 64 --data-type uint8 --nodata 0 -y >> logs/04_export_detalhe.log 2>&1
  for i in 1 2 3; do [ -s "$OUT/_failures.json" ] || break; "$DS" retry -o "$OUT" >> logs/04_export_detalhe.log 2>&1; done
  n=$(ls "$OUT"/*.tif 2>/dev/null | wc -l); f=$( [ -s "$OUT/_failures.json" ] && wc -l < "$OUT/_failures.json" || echo 0)
  echo "EXPORT $R | tiles=$n | falhas=$f | $(( $(date +%s)-t0 ))s | $(du -sh "$OUT" | cut -f1)" | tee -a "$LOG"
}
if [ -n "$1" ]; then um "$1"; exit; fi
while read -r R || [ -n "$R" ]; do R=$(echo "$R" | tr -d '\r'); [ -n "$R" ] && um "$R"; done < dados/aoi/regioes.txt
echo "EXPORT_FIM $(date '+%F %T') | total $(du -sh dados/embeddings | cut -f1)" | tee -a "$LOG"
