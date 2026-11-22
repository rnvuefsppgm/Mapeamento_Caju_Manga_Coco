#!/usr/bin/env bash
# Etapa de DADOS (sem treino): espera a exportação (04) -> VRT (05) -> MapBiomas (06) -> rótulos campo (07) -> chips (08) -> cache (09)
cd "$(dirname "$0")"; PY=D:/Modelo_Caju/.venv/Scripts/python.exe; LOG=logs/run_dados.log
until grep -q EXPORT_FIM logs/04_export.log 2>/dev/null; do sleep 60; done
step() { echo "=== $(date '+%F %T') | $*" >> "$LOG"; "$PY" "$@" >> "$LOG" 2>&1; rc=$?; echo "=== rc=$rc | $1" >> "$LOG"; return $rc; }
step 05_vrt.py && step 06_mapbiomas.py && step 07_rotulos_campo.py && step 08_chips.py && step 09_cache.py denso \
  && echo "DADOS_FIM $(date '+%F %T')" >> "$LOG" || echo "DADOS_ERRO $(date '+%F %T')" >> "$LOG"
grep -h "RESUMO\|VRT:" "$LOG" | tail -8
