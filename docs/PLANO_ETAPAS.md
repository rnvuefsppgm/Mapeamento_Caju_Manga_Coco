# Plano de etapas — Mapa de coco 2025

Dados: `D:\Modelo_Coco` (`dados_externos/` compartilhado; `v1/` primeira versão). Código: `scripts/`.
**Ponto de parada combinado: fim da E4 (antes do treino), com revisão do usuário.**

| # | Etapa | Scripts | Entrada → Saída | Quem | Situação |
|---|---|---|---|---|---|
| E0 | Organização + economia de tokens | `CLAUDE.md`, `STATUS.md`, `tools/resumo.py` | – | Claude | ✅ |
| E1a | PAM 2023–25 (SIDRA 1613, coco 2727) | `00_pam_coco.py` | → `pam/pam_coco_municipios.csv`, rankings | Claude | ✅ |
| E1b | Censo Agro 2017 (SIDRA 6955, coco 4945) | `01_censo_agro_coco.py` | → `censo_agro/…csv`; junção `pam/pam_censo_coco.csv` | Claude | ✅ |
| E1c | Polos do CE (Neves et al. 2022, Fig. 1) | `config.yaml: polos_ce` | estratos de amostragem no CE | Claude | ✅ (aprox.) |
| E1d | Municípios do recorte final | `02_municipios.py` (a fazer) | ranking PAM × Censo → `municipios_coco.gpkg` | Claude → **usuário confirma** | pendente |
| E2 | Embeddings 2025 | `03_regioes_export.py <lista>`, `run_export.sh <lista>` | regiões (UTM local) → tiles COG uint8 64 bandas | Claude | 🔄 lista `amostragem_inicial` |
| **E3a** | **Amostras-modelo do usuário** | Google Earth Pro, pasta `COCO` → KML em `dados_externos/amostras_usuario/` | 2 polígonos (Paraipaba) | **usuário** | ✅ coletadas / ⏳ salvar KML |
| **E3b** | **Candidatos por similaridade** (Paraipaba + top 5 de produção) | `05_candidatos_similaridade.py` | kNN cosseno nos embeddings → `candidatos_coco.kml` (alta/média/limiar + controle negativo) | Claude | pronto p/ rodar |
| **E3c** | **Conferência no Google Earth** | usuário move candidatos para `COCO_ok` / `NAO_COCO` / `DUVIDA` e salva KML | → `r_ouro` v0 | **usuário** | – |
| E3d | Rodadas seguintes (aprendizado ativo) | `05_…` com R-ouro v0 como novo modelo + `11_amostra_fotointerpretacao.py` (estratos S1–S6 + polos CE) | expande para os 25 municípios | Claude + usuário | – |
| E3e | Fontes prévias | `10_fontes.py` (de caju v10/01) + Descals 2023 | XGB PU, MapBiomas, caju v10b como negativo | Claude | – |
| E3f | Rótulos automáticos + QC | `13_rotulos_concordancia.py`, `14_qc_rotulos.py` | raster {0,1,255}; 200 pontos QC | Claude + usuário | trava: precisão coco ≥ 0,90 |
| E4 | Chips + cache | `20_chips.py`, `21_cache.py` (de caju v10b/03, 02c) | chips 256 px uint8 + memmap | Claude | **PARADA: revisão** |
| E5 | Treino | `30_cv_espacial.py` … `33_fusao_xgb.py` | CV 5 folds só em R-ouro; final 5 sementes × 100 ép. a partir dos pesos do caju | Claude | trava IoU ≥ 0,65 |
| E6 | Predição + mosaico | `40_predicao.py`, `41_mosaico.py` | COG único classe/prob | Claude | – |
| E7 | Validação e comparação | `50_comparacao.py`, `51_validacao_independente.py` | mapa × PAM × Censo × MapBiomas × Descals; amostra estratificada (Olofsson) | Claude + usuário | – |

## Municípios da amostragem inicial (produção média PAM 2023–25)
| # | Município | Produção (mil frutos) | Área destinada PAM (ha) | Área existente Censo 2017 (ha) | UTM |
|---|---|---|---|---|---|
| 1 | Paraipaba-CE | 161.017 | 6.109 | 2.651 | 24S |
| 2 | Petrolândia-PE | 152.833 | 3.717 | 1.652 | 24S |
| 3 | Acaraú-CE | 111.668 | 6.806 | 4.499 | 24S |
| 4 | Rodelas-BA | 83.415 | 2.357 | 2.370 | 24S |
| 5 | Moju-PA | 81.128 | 7.733 | 6.351 | 22S |
| 6 | Trairi-CE | 77.303 | 8.282 | 4.485 | 24S |
Paraipaba já é o 1º; Trairi (6º) completa "Paraipaba + top 5". Petrolândia/Rodelas = coco irrigado do São Francisco; Moju = coco amazônico (paisagem distinta).

## Decisões em aberto
1. Recorte final: top 25 Brasil (51% da área; inclui PA/ES, 3 fusos) ou Nordeste (57%). PAM e Censo divergem (PAM 188,9 mil ha × Censo 152,8 mil ha).
2. Autorizar download do mapa global de coqueiro (Zenodo, Descals 2023).
