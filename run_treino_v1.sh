#!/usr/bin/env bash
# v1 (Chapada): CV A (caju sem cabeça) e B (ImageNet) -> limiar OOF -> escolhe o melhor -> trava IoU -> final 5 sementes
#               -> predição 25 municípios -> mosaico -> comparação. Retomável (pula CV já feito). Log: logs/run_treino_v1.log
cd "$(dirname "$0")"; PY=D:/Modelo_Caju/.venv/Scripts/python.exe; LOG=logs/run_treino_v1.log; R=$(pwd)
export PYTHONIOENCODING=utf-8 GPU_AUG=1
step() { echo "=== $(date '+%F %T') | $*" >> "$LOG"; "$PY" "$@" >> "$LOG" 2>&1; rc=$?; echo "=== $(date '+%F %T') | rc=$rc | $1" >> "$LOG"; return $rc; }
for X in A_caju_sem_cabeca B_imagenet; do
  N=v1_${X%%_*}
  [ -f reports/spatial_cv_$N.json ] && continue
  FT_INIT="$R/modelos/init/$X.pt" step 13_cv.py --model cnn --folds 5 --epochs 60 --patience 15 --channels both \
      --name $N --meta chips_metadata_v1.parquet || exit 1
done
step 14_limiar_oof.py v1_A v1_B || exit 1
read BEST IOU TH < <("$PY" -c "
import json;r={k:json.load(open(f'reports/oof_v1_{k}.json')) for k in 'AB'};b=max(r,key=lambda k:r[k]['iou'])
print(b,round(r[b]['iou'],4),r[b]['limiar'])")
INIT=$([ "$BEST" = A ] && echo A_caju_sem_cabeca || echo B_imagenet)
echo "=== ESCOLHA: $BEST ($INIT) IoU=$IOU limiar=$TH" | tee -a "$LOG"
"$PY" -c "import sys;sys.exit(0 if $IOU>=0.60 else 1)" || { echo "=== V1_TRAVA_FALHOU IoU=$IOU" | tee -a "$LOG"; exit 2; }
ls modelos/final_seed*.pt >/dev/null 2>&1 && [ $(ls modelos/final_seed*.pt | wc -l) -ge 5 ] || step 15_final.py "$R/modelos/init/$INIT.pt" "$TH" || exit 1
step 16_predicao.py "$TH" || exit 1
step 17_mosaico.py
step 18_comparacao.py
echo "=== V1_FIM $(date '+%F %T')" | tee -a "$LOG"
