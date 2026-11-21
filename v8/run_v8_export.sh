#!/usr/bin/env bash
# v8 — exporta embeddings 2025 (DatensEE, sem tarefas EE) para todas as regiões de data/aoi/regions.txt. Retomável.
cd "$(dirname "$0")"
while read -r R || [ -n "$R" ]; do
  R=$(echo "$R" | tr -d '\r'); [ -z "$R" ] && continue
  D=data/ee_exports/2025/embeddings/$R
  if ls "$D"/*.tif >/dev/null 2>&1 && [ ! -s "$D/_failures.json" ]; then echo "EXPORT pulado $R (já existe)" >> logs/export.log; continue; fi
  ./export_one.sh embeddings "$R"
done < data/aoi/regions.txt
echo "EXPORT_FIM $(date '+%F %T') | total $(du -sh data/ee_exports/2025/embeddings | cut -f1)" | tee -a logs/export.log
