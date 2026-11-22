# Desempenho — ajustes de velocidade (sem mudar o método)

| Data | Script | Antes | Mudança | Efeito esperado | Impacto na qualidade |
|---|---|---|---|---|---|
| 21/11/2026 | `05_candidatos_similaridade.py` | modelo com todos os pixels (≈ 400 mil no SF) | modelo limitado a 3.000 px (sorteio estratificado por polígono) | cabe na GPU de 4 GB | desprezível (k-NN k=5 sobre 3.000 px representativos) |
| 21/11/2026 | `13_xgb_rotulos.py` (amostragem) | fontes (MapBiomas, Descals, caju) lidas e vetores reprojetados/rasterizados **a cada tile**; todos os tiles amostrados | fontes e vetores **uma vez por região**; amostra todos os tiles com positivos + 40 % aleatórios dos demais (`rotulos.frac_tiles_negativos`) | 1ª passada ~3 s/tile → ~1 s/tile e ~50 % menos tiles: **~1 h → ~20 min** nos 15 municípios | nenhum (fontes/vetores) / muito pequeno (negativos já sobram: ~150 mil × poucos mil positivos; limite de 400/tile) |

Observações
- A 2ª passada (predição) roda em todos os tiles; não muda.
- A rodada v1 dos rótulos (iniciada às ~15:20 de 21/11) usou a versão **antes** da otimização; vale a partir da próxima execução.
- Concorrência na GPU/CPU (predição do caju, outra busca por similaridade) aumenta os tempos; medir com a máquina livre.
