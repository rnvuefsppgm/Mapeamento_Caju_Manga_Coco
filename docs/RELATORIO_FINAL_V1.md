# Mapa de coco 2025 — v1 (top 15 Caatinga NE) — 22/11/2026

**Produtos** (`D:\Modelo_Coco\v1\predictions\`): `coco_2025_top15_caatinga_classe.tif` (0/1, 255 = fora) e `_prob.tif` (0–100),
EPSG:32724, 10 m; por município em `predictions/2025/`. Comparação completa: `v1/reports/comparacao_final.md/csv`.

## Treino
| Item | Valor |
|---|---|
| CV espacial 5 folds (blocos 10 km) | IoU OOF **0,789**, F1 0,882, precisão 0,891, recall 0,873, limiar 0,65 → trava (0,65) **aprovada** |
| IoU por fold | 0,577 · 0,721 · 0,850 · 0,873 · 0,872 |
| Só ouro (~1 mil px coco/fold) | IoU 0,088, precisão 0,094, recall 0,545 — pouco ouro; não confiável |
| Final | U-Net resnet18, 5 sementes × 100 épocas (≈ 36 min cada), ensemble + TTA ×2 |

## Área (ha) — mapa × estatística
Total: mapa **23.247** | PAM 54.136 | Censo 33.325 | MapBiomas 48: 212.423 | Descals 1.865. Correlação mapa × PAM 0,34.

| Grupo | Municípios | Razão mapa/PAM |
|---|---|---|
| Bom (0,5–1,3) | Touros, Paraipaba, Petrolândia, Rodelas, Petrolina, Juazeiro | 0,62–1,30 |
| Médio | Casa Nova | 0,46 |
| Subestimado (≤ 0,10) | Trairi, Acaraú, Itarema, Itapipoca, Beberibe, Amontada, Cascavel, Camocim | 0,00–0,10 |

## Limitações
1. **Litoral CE fora de Paraipaba subestimado** — reflete os rótulos (cobertura ≤ 0,03 da área de referência nesses municípios). O
   modelo aprendeu o coqueiral irrigado em grade (Paraipaba, Touros, São Francisco) e não o coqueiral de sequeiro/disperso.
2. Touros: mapa 6.259 ha > Censo 2.814 (≈ PAM 6.633); MapBiomas 48 quase ausente lá → conferir visualmente.
3. Métrica "ouro sobre automático" herda os rótulos automáticos; precisão real só com mais pontos ouro.

## Recomendação (rodada v2)
Coletar amostras de coqueiral de sequeiro/disperso no litoral CE (Trairi, Acaraú, Itarema, Itapipoca, Amontada, Beberibe,
Cascavel, Camocim) — fotointerpretação nos candidatos do 05 e conferência de pontos (15) — e retreinar (≈ 4 h).
