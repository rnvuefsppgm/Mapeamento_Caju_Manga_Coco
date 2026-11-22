# Plano — Mapa de manga 2025 (estratégia do caju v10b)

Pasta: `D:\Backup_2026\Geral_desktop\MAPA_MANGA` · Base: `D:\Modelo_Caju\v10b` (+ v8, v10, v10a de onde ela herda).
Regras de trabalho e economia de tokens: `CLAUDE.md`. Decisões: `DECISOES.md`. Estado: `python 00_status.py`.

## Situação de partida
| Item | Caju | Manga |
|---|---|---|
| Amostras de campo | Serra do Mel (RN), 466 polígonos, ~18 mil ha (PAM 38 mil) | Livramento de N. Sra. (BA), **258 polígonos únicos**, **1.683 ha** (PAM 2025: 6.630 ha), área de 5,9 × 11,3 km |
| Municípios | 25 maiores da Caatinga (PAM 2023–25) | 25 maiores da Caatinga (PAM 2023–25) = **92,5%** da manga do bioma; 77 mil km² com buffer |
| Dados | Embeddings AlphaEarth 2025 (64 bandas, 10 m) | idem |
| MapBiomas Col. 11 nas amostras | 66% em 41, 19% em 48 | **84% em Pastagem (15)**, 7% em Café (46), 6% em Outras perenes (48) |

## Etapas
| # | Etapa | Scripts | Caju equivalente | Situação |
|---|---|---|---|---|
| **A. Dados** (sem treino) |||||
| A1 | Amostras KML/KMZ → GeoPackage (dedup, área, município) | `01_amostras.py` | – | feito |
| A2 | PAM manga 2023–25 (SIDRA 1613/2737) + seleção dos 25 municípios | `02_pam_municipios.py` | `dados_externos/ranking_*` | feito |
| A3 | Regiões de exportação (município + 1 km) | `03_regioes.py` | v8 `01_build_expressions` | feito |
| A4 | Exportação dos embeddings 2025 (DatensEE, ~65 GB, ~2 h) | `04_export_embeddings.sh` | v8 `export_one.sh` | feito (25 municípios, 0 falhas, 38 GB, 45 min) |
| A5 | VRT por município | `05_vrt.py` | v8 `03_mosaic_vrt` | feito |
| A6 | MapBiomas 2025 recortado a 10 m + classes nas amostras | `06_mapbiomas.py` | v10b `30` | feito (25 recortes) |
| A7 | Raster de rótulos de campo em códigos | `07_rotulos_campo.py` | v9/v10b `02_chips_v9` | feito |
| A8 | Chips de campo (uint8) | `08_chips.py` | v10b `03_chips` | feito |
| A9 | Cache 8 bits | `09_cache.py` | v10b `02c` + conversão u8 | feito |
| ⏸ | **Parada: revisão com o usuário antes do treino** ||||
| **B. Modelo base** (= caju v7/v8 + fontes da v10) |||||
| B1 | U-Net resnet18 só embeddings, CV espacial 5 folds (blocos de 3 km), limiar fora do fold | `10_cv_base.py` (porte do `08_spatial_cv`) | v7 C1 | – |
| B2 | XGBoost por pixel na GPU (PU: positivos = polígonos; negativos confiáveis) | `11_xgb.py` (porte do v10 `01_fontes`) | v10 S2 | – |
| B3 | Predição base nos 25 municípios + comparação com a PAM | `12_predicao_base.py` | v8 `04b` | – |
| **C. Pseudo-rótulos** (= caju v10 → v10a → v10b, numa passada) |||||
| C1 | XGBoost regional (3 regiões: Vale do São Francisco, Chapada/Brumado-BA, Norte de MG + RN/CE) | `21_xgb_regional.py` | v10a `21` | – |
| C2 | Rótulos por ranking com prior PAM (manga ≤ 0,6× PAM; não-manga só abaixo de 3× PAM) | `22_rotulos_rank.py` | v10a `22` | – |
| C3 | Máscara MapBiomas: fora de C = {15, 46, 48, 41, 21} → não-manga; anel de 2 px ignorado | `23_rotulos_mb.py` | v10b `31` | – |
| C4 | Chips por município (30 com manga + 30 sem) + cache | `24_chips_pseudo.py` | v10b `03` + `02c` | – |
| **D. Modelo final** |||||
| D1 | CV com ajuste fino a partir do modelo base + trava de IoU | `30_cv_ft.py` | v10b `08` + `10b` | – |
| D2 | Fusão com o XGB regional (só se ganhar no OOF) | `31_fusao.py` | v10b `17` | – |
| D3 | Treino final 100 épocas × 5 sementes (aquecimento 5, encoder lento 10, EMA, + Lovász) | `32_final.py` | v10b `14` | – |
| D4 | Predição 25 municípios (TTA ×2, pós-processamento 1 ha) + raster único + comparação PAM/MapBiomas | `33_predicao.py`, `34_mosaico.py`, `35_comparacao.py` | v10b `15, 32, 16` | – |

## Pontos a decidir antes do treino
1. **Negativos do campo** (`09_cache.py denso|mb`): *denso* trata a área amostrada (envoltória + 500 m) como mapeada por completo (como o caju v7); *mb* usa só negativos MapBiomas (como o caju v10b). Proposta: CV com os dois e ficar com o melhor no OOF.
2. **Poucos chips de campo**: 110 chips (92 com manga) com stride 96, contra 891 no caju. Proposta: stride 64 (~2,3×) para o modelo base.
3. **Blocos do CV**: as amostras cabem em ~6 × 11 km; com blocos de 10 km (caju) haveria só 2–4 blocos. Proposta: 3 km.
4. **Municípios do MG** (Jaíba, Matias Cardoso, Janaúba) ficam no fuso 23; tudo segue em EPSG:32724 como no caju (erro de área < 1%, corrigido na contagem final).
5. **Municípios grandes** (Sento Sé 12,6 mil km², Casa Nova 9,7 mil km²) entram inteiros, como no caju.

## Riscos
- Amostras só numa paisagem (Chapada/Brumado); o Vale do São Francisco (Petrolina/Juazeiro, ~70% da produção) é irrigado e diferente → dependência forte dos pseudo-rótulos (etapa C), como foi no caju.
- Manga × outras frutíferas irrigadas (uva, coco, banana, goiaba) no Vale; a MapBiomas não separa.
