#!/usr/bin/env bash
# Encadeia E5→E7 sem intervenção (plano em docs/PLANO_NOTURNO.md). Retomável: cada etapa pula o que já existe.
# 0) espera a GPU liberar (treino da manga e outros) → 1) CV 5 folds → 2) limiar OOF + trava → 3) final 5 sementes
# → 4) predição 15 municípios → 5) mosaico único → 6) comparação. Log de marcos: v1/logs/run_treino_mapa.log
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
cd "$(dirname "$0")"
PY=D:/Modelo_Caju/.venv/Scripts/python.exe; L=D:/Modelo_Coco/v1/logs; R=D:/Modelo_Coco/v1/reports; LISTA=top15_caatinga
marco() { echo "$(date '+%F %H:%M') | $*" | tee -a $L/run_treino_mapa.log; }
# Espera a manga terminar POR COMPLETO (run_pos_treino_v1.sh: predição→mosaico→comparação→KMZ) e qualquer 30_treino lançado por ela.
ocupada() { powershell -NoProfile -ExecutionPolicy Bypass -File gpu_ocupada.ps1; }

marco "0 aguardando GPU (manga/outros)"; while ocupada; do sleep 120; done; marco "0 GPU livre"
marco "1 CV início"; $PY -W ignore 30_treino.py cv 60 15 >> $L/30_cv.log 2>&1 || { marco "1 CV FALHOU"; exit 1; }
marco "2 limiar"; $PY -W ignore 31_limiar_oof.py > $L/31_limiar.log 2>&1 || { marco "2 limiar FALHOU"; exit 1; }
APROV=$($PY -c "import json;print(json.load(open('$R/cv_resumo.json'))['aprovado'])")
marco "2 CV: $(head -3 $R/cv_resumo.md | tail -1 | cut -c1-200)"
# Decisão pré-acordada: trava reprovada NÃO interrompe — segue e o mapa sai marcado como PRELIMINAR no relatório.
marco "3 final início (trava aprovada=$APROV)"
for s in 42 7 123 2024 99; do
  [ -f D:/Modelo_Coco/v1/models/final_seed$s.pt ] && continue
  $PY -W ignore 30_treino.py final 100 $s >> $L/30_final.log 2>&1 || { marco "3 final seed $s FALHOU"; exit 1; }
  marco "3 final seed $s ok"
done
marco "4 predição"; $PY -W ignore 40_predicao.py $LISTA > $L/40_predicao.log 2>&1 || { marco "4 predição FALHOU"; exit 1; }
marco "5 mosaico"; $PY -W ignore 41_mosaico.py $LISTA > $L/41_mosaico.log 2>&1 || { marco "5 mosaico FALHOU"; exit 1; }
marco "6 comparação"; $PY -W ignore 50_comparacao.py $LISTA > $L/50_comparacao.log 2>&1 || { marco "6 comparação FALHOU"; exit 1; }
marco "FIM — mapa: v1/predictions/coco_2025_${LISTA}_classe.tif | relatório: v1/reports/comparacao_final.md"
