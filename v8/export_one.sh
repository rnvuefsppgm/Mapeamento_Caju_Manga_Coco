#!/usr/bin/env bash
# uso: ./export_one.sh <produto: embeddings|sentinel> <regiao_id>   (ambiente .venv_v8)
cd "$(dirname "$0")"; DS=../.venv_v8/Scripts/datensee; P=$1; R=$2
case $P in embeddings) NB=64; DT=uint8; ND=0;; *) echo "v8: so embeddings (Sentinel em ../v8_sentinel)"; exit 1;; esac
OUT=data/ee_exports/2025/$P/$R; mkdir -p "$OUT"
t0=$(date +%s)
$DS export expressions/${P}_2025.json data/aoi/regions/$R.geojson -p rnv-geodatin -o "$OUT" \
   --scale 10 --crs EPSG:32724 --tile-size 512 --band-count $NB --data-type $DT --nodata $ND -y >> logs/export.log 2>&1
for i in 1 2 3; do [ -s "$OUT/_failures.json" ] || break; $DS retry -o "$OUT" >> logs/export.log 2>&1; done
n=$(ls "$OUT"/*.tif 2>/dev/null | wc -l); f=$( [ -s "$OUT/_failures.json" ] && wc -l < "$OUT/_failures.json" || echo 0)
echo "EXPORT $P $R | tiles=$n | falhas=$f | $(( $(date +%s)-t0 ))s | $(du -sh "$OUT" | cut -f1)" | tee -a logs/export.log
