# Plano de execução autônoma — treino ao mapa final (21–22/11/2026)

Autorização do usuário: rodar todas as etapas até o mapa de coco dos 15 municípios, com decisões plausíveis
dentro do que foi pedido. Encadeador: `scripts/run_treino_mapa.sh` (retomável). Marcos: `v1/logs/run_treino_mapa.log`.

| # | Etapa | Script | Tempo estimado (GPU livre) | Saída |
|---|---|---|---|---|
| 0 | Esperar o treino final da manga (`15_final.py`, MAPA_MANGA) liberar a GPU | — | depende da manga | — |
| 1 | CV espacial 5 folds (até 60 épocas, paciência 15; blocos de 10 km) | `30_treino.py cv` | 2–4 h | `v1/models/cv/fold*.pt`, `v1/reports/oof_fold*.npz` |
| 2 | Limiar global fora do fold + trava IoU ≥ 0,65 | `31_limiar_oof.py` | 1 min | `v1/reports/cv_resumo.md/json` |
| 3 | Modelo final: 5 sementes (42, 7, 123, 2024, 99) × 100 épocas, todos os chips, partindo dos pesos do caju | `30_treino.py final` | 5–7 h | `v1/models/final_seed*.pt` |
| 4 | Predição 15 municípios: 5 membros + TTA ×2, blocos 1024 + margem 64, limiar da CV, MapBiomas seguras zeradas, manchas ≥ 1 ha | `40_predicao.py` | 1,5–2 h | `v1/predictions/2025/<município>_{prob,classe}.tif` |
| 5 | Raster único (EPSG:32724; Touros reprojetado) | `41_mosaico.py` | 10 min | `v1/predictions/coco_2025_top15_caatinga_{classe,prob}.tif` |
| 6 | Comparação mapa × PAM × Censo × MapBiomas 48 × Descals × rótulos | `50_comparacao.py` | 5 min | `v1/reports/comparacao_final.md/csv` |
Total após a GPU liberar: **~9–13 h**.

## Decisões pré-acordadas (sem perguntar)
1. **Trava reprovada (IoU < 0,65)**: segue até o mapa; o relatório marca o mapa como **preliminar** e recomenda a rodada 2 de amostras no litoral.
2. **Seleção de época** na CV por IoU "ouro sobre automático" (o "só ouro" tem ~1 mil px de coco por fold); os dois são reportados.
3. **Épocas do final** = 100 fixas (receita caju v10b), mesmo que a CV pare antes.
4. **Falha de um script**: o encadeador para e registra; na retomada as etapas prontas são puladas. Se for erro simples de código, o Claude corrige e retoma.
5. **GPU ocupada de novo por outro projeto no meio do caminho**: segue dividindo (não interrompe processos de outros projetos).
6. **Nenhum download, nenhuma alteração fora de `Mapa_COCO` e `D:\Modelo_Coco`.**

## Ao terminar
Atualizar `STATUS.md`, `DECISOES.md` e `docs/RELATORIO_PRE_TREINO.md` → resumo final com métricas da CV, área por município × PAM/Censo e limitações.
