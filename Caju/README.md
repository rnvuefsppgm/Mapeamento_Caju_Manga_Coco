# Mapeamento de Caju na Caatinga

Mapeamento de pomares de caju (*Anacardium occidentale*) na Caatinga com **deep learning (U-Net)** e **embeddings de satélite AlphaEarth** (Google Satellite Embedding, 64 dimensões, 10 m).

> **Onde fica:** esta é a pasta `Caju/` do repositório geral [Mapeamento_Caju_Manga_Coco](https://github.com/rnvuefsppgm/Mapeamento_Caju_Manga_Coco) (caju, manga, coco e mapa único das culturas). Novas versões do caju entram aqui. O repositório antigo [Mapeamento_Caju](https://github.com/rnvuefsppgm/Mapeamento_Caju) guarda o histórico até o commit 36ce89f e não recebe mais atualizações.

O produto atual é o **mapa de caju de 2025 dos 25 maiores municípios produtores da Caatinga**. Esses 25 municípios somam 67,6% da área colhida e 75,8% da produção de castanha de caju da Caatinga nordestina (PAM/IBGE 2025).

| | |
|---|---|
| **Mapa final** | [`resultados/caju_2025_25mun_FINAL_classe.tif`](resultados/) — 0 = não-caju, 1 = caju, 255 = fora da área. EPSG:32724, 10 m, COG |
| **Área mapeada** | 248.513 ha (PAM 2025: 298.195 ha) |
| **Concordância com a PAM** | correlação 0,95; 21 de 25 municípios com razão mapa/PAM entre 0,5 e 2,5 |
| **Validação cruzada espacial** (Serra do Mel, fora do fold) | IoU 0,804 · F1 0,891 · precisão 0,904 · recall 0,879 |

---

## Sumário
1. [Como o método funciona](#1-como-o-método-funciona)
2. [Estrutura do repositório](#2-estrutura-do-repositório)
3. [Instalação](#3-instalação)
4. [Dados de entrada (não incluídos)](#4-dados-de-entrada-não-incluídos)
5. [Tutorial: sequência completa de scripts](#5-tutorial-sequência-completa-de-scripts)
6. [Atalho: só gerar o mapa com modelos já treinados](#6-atalho-só-gerar-o-mapa-com-modelos-já-treinados)
7. [Resultados](#7-resultados)
8. [Limitações conhecidas](#8-limitações-conhecidas)
9. [Documentação detalhada](#9-documentação-detalhada)

---

## 1. Como o método funciona

```
Embeddings AlphaEarth 2025 (64 bandas, 10 m)
        │
        ├─► U-Net (resnet18) treinada nos polígonos de campo de Serra do Mel (RN) ............ v7
        │
        ├─► Download dos 25 municípios (DatensEE, sem tarefas no Earth Engine) + 1º mapa ....... v8
        │
        ├─► Amostras por concordância: U-Net × XGBoost × agrupamento ........................... v10
        │
        ├─► XGBoost regional (3 regiões) + rótulos por ranking com prior de área da PAM ......... v10a
        │
        └─► Máscara MapBiomas 2025 + ajuste fino da U-Net (100 épocas × 5 sementes) ........... v10b
                → predição com TTA + fusão regional → mapa final (opção c)
```

**Ideias principais:**
- **Só embeddings.** Os embeddings AlphaEarth já resumem a série temporal de Sentinel-1/2 e Landsat do ano. O modelo usa só eles (64 canais), sem baixar imagens brutas.
- **Poucos dados de campo.** Os polígonos de campo existem só em Serra do Mel (RN) e cobrem ~18 mil dos 38 mil ha da PAM. As amostras dos outros municípios são geradas de forma automática e conservadora (v10 → v10b).
- **Rótulos com três estados.** Caju, não-caju confiável e *ignorado*. O que não é certo fica fora da perda e da validação, então o caju real fora dos polígonos não é tratado como "não-caju".
- **Máscara MapBiomas (Coleção 11, 2025).** Só 19% do caju de campo cai na classe "lavoura perene" da MapBiomas; 66% cai em "lavoura temporária". Por isso as classes compatíveis com caju são {48, 41, 15, 21}, que contêm 98,6% dos polígonos de campo. O que fica fora delas vira não-caju.
- **Validação espacial honesta.** CV de 5 folds em blocos de 10 km, limiar global escolhido nas predições fora do fold e ajuste fino fold a fold, sem vazamento.
- **Tudo na GPU** (testado numa RTX 3050 de 4 GB): treino com AMP; normalização e aumento de dados na GPU (`GPU_AUG=1`); XGBoost com `device="cuda"`; predição com TTA.

---

## 2. Estrutura do repositório

As pastas `vN` são **etapas encadeadas**: cada versão usa saídas da anterior por caminho relativo (`../vN/...`). Mantenha a estrutura e rode os comandos de dentro de `Caju/`.

```
Caju/                        (pasta do repositório Mapeamento_Caju_Manga_Coco)
├── README.md                ← este tutorial
├── requirements.txt
├── dados_externos/          PAM/IBGE (castanha de caju 2023–2025), ranking da Caatinga, biomas, limites dos 25 municípios
├── docs/                    DECISOES.md (registro de decisões) e VERSOES_SINTESE.md
├── resultados/              mapa final (classe e probabilidade), áreas e comparações
├── v7/   base do código (src/), download do treino (Sentinel + embeddings), CV da U-Net "C1"
├── v8/   embeddings 2025 dos 25 municípios via DatensEE + 1ª predição na GPU
├── v9/   estatísticas MapBiomas pela API (lavoura perene), chips de campo com faixa de exclusão
├── v10/  fontes independentes (XGBoost na GPU, similaridade, k-means) + rótulos por concordância + modelo v10
├── v10a/ XGBoost regional + rótulos por ranking (prior PAM) + utilitários de ajuste fino (src/ft.py, src/pred_a.py)
└── v10b/ máscara MapBiomas + ajuste fino final + mapa final  ← versão atual
```

Cada versão tem `src/` (módulos), `config/config.yaml`, scripts numerados na ordem de execução, `run_*.sh` (fluxo completo) e um `.md` com o que mudou.

---

## 3. Instalação

**Requisitos:** Windows ou Linux, Python 3.11, GPU NVIDIA com CUDA 13 (≥ 4 GB) e bash (no Windows, o Git Bash).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |  Linux: source .venv/bin/activate
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
earthengine authenticate          # conta com acesso a um projeto Google Cloud registrado no Earth Engine
```

**Ambiente separado para o DatensEE (v8):** exige Python ≥ 3.12 e Java 17+.
```bash
python3.14 -m venv .venv_v8 && .venv_v8/Scripts/pip install datensee==0.1.0a8 earthengine-api
```
O DatensEE usa a credencial padrão do Google Cloud (`gcloud auth application-default login`).

**Ajuste os caminhos.** Cada `vN/config/config.yaml` tem `output_root` (e `paths.vectors`) apontando para `D:/Modelo_Caju/vN`. Troque pelo caminho onde clonou o repositório, ou clone em `D:/Modelo_Caju`. No `v7/config/config.yaml`, troque também `embeddings.project` pelo ID do seu projeto do Earth Engine, assim como o `-p` de `v8/export_one.sh`.

Os scripts `run_*.sh` usam `../.venv/Scripts/python.exe`, o caminho no Windows. No Linux, troque por `../.venv/bin/python`.

---

## 4. Dados de entrada (não incluídos)

| Dado | Onde colocar | Como obter |
|---|---|---|
| **Polígonos de campo de caju** (Serra do Mel) | `v7/data/vectors/Pol_Caju.shp` | Dado próprio, não público. Substitua pelos seus polígonos de referência (campo `geometry`; tudo dentro = caju) |
| **Limites municipais** | `v7/data/vectors/municipios.shp`; os 25 municípios já estão em `dados_externos/municipios25/` | IBGE, malha municipal 2024 |
| **MapBiomas Coleção 11, 2025 (30 m)** | qualquer pasta; informe em `MB_RASTER` | Plataforma MapBiomas (download do GeoTIFF do Brasil). Padrão: `D:/Eduarda/raster_new/brazil_coverage-col11_2025.tif` |
| Embeddings AlphaEarth 2025 | baixados pelos scripts (v7 `03a`, v8 `run_v8_export.sh`) | Earth Engine: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` (~2 GB nos 25 municípios em uint8) |
| PAM/IBGE castanha de caju | já em `dados_externos/` | SIDRA, tabela 1613 |

Modelos (`.pt`), caches e chips **não** estão no repositório por causa do tamanho. Eles são recriados pelos scripts.

---

## 5. Tutorial: sequência completa de scripts

> Rode cada script **de dentro da pasta da versão** (os caminhos são relativos a ela). Cada etapa grava em `vN/logs/`, `vN/reports/`, `vN/models/` e `vN/predictions/`. Os `run_*.sh` encadeiam as etapas e param em erro.

### Etapa 1 — v7: dados da área de treino e U-Net base ("C1")
```bash
cd v7
python 01_download_training_data.py --vector Pol_Caju.shp --role ground_truth   # grade Sentinel da área de treino
python 03a_download_embeddings.py --project SEU_PROJETO                         # embeddings 2025 na mesma grade (sem tarefas EE)
python 02_create_training_chips.py                                               # chips 256×256 + rótulos de campo
python 02c_build_cache.py                                                        # cache memmap
python 08_spatial_cv.py --model cnn --folds 5 --epochs 60 --patience 15 --channels emb --name unet_emb
```
Saída principal: `v7/models/ensemble_unet_emb/fold1-5.pt` (modelo C1, usado na 1ª predição).

### Etapa 2 — v8: embeddings 2025 dos 25 municípios e 1º mapa
```bash
cd ../v8
python 01_build_expressions.py          # expressão EE (uint8, valor*127+128) + regiões (25 municípios + buffer 1 km)
./run_v8_export.sh                      # DatensEE, município a município (retomável); usa export_one.sh e .venv_v8
python 03_mosaic_vrt.py                 # VRT por município
python 04b_predict_emb_gpu.py           # predição na GPU com o C1 → predictions/2025/<mun>_prob.tif
python 05_compare_pam.py                # área × PAM
```

### Etapa 3 — v9: MapBiomas pela API (estatísticas)
```bash
cd ../v9
python 13_mapbiomas_perene.py           # área de lavoura perene 2025 por município → reports/mapbiomas_perene_2025.csv
```

### Etapa 4 — v10: amostras por concordância e modelo v10
```bash
cd ../v10
python 01_fontes.py                     # XGBoost (GPU), similaridade e k-means por região → data/fontes/
./run_v10.sh                            # 02_rotulos → 03_chips → 02c → 11_oof_v7 → 08 (CV) → 10b → 14 (final) → 15 (predição) → 16
```
Saídas usadas adiante: `v10/models/ensemble_unet_conc/fold*.pt`, `v10/models/final_seed{42,7,123}.pt` e `v10/predictions/2025/*_prob.tif`.

### Etapa 5 — v10a: XGBoost regional e rótulos por ranking
```bash
cd ../v10a
python 21_xgb_regional.py               # XGBoost por região, treinado nos municípios "bons" da v10 → data/fontes_a/<mun>_xr.tif
python 22_rotulos_rank.py               # rótulos por ranking: caju = topo até 0,6×PAM; não-caju = abaixo de 3×PAM → data/rotulos/
```
(Os passos seguintes da v10a, `run_v10a.sh` e `run_v10a_ft.sh`, são opcionais. A v10b usa só os rótulos e o XGBoost regional.)

### Etapa 6 — v10b: máscara MapBiomas, ajuste fino final e mapa final ✅
```bash
cd ../v10b
export MB_RASTER=/caminho/brazil_coverage-col11_2025.tif
./run_v10b.sh
```
O `run_v10b.sh` executa, nesta ordem:

| # | Script | O que faz | Tempo (RTX 3050) |
|---|---|---|---|
| 1 | `30_mapbiomas_recorte.py` | Recorta e reamostra a MapBiomas 2025 para a grade de 10 m de cada município; mede as classes dentro do caju | ~5 min |
| 2 | `31_rotulos_mb.py` | Rótulos v10a + máscara MapBiomas: fora de {48, 41, 15, 21} = não-caju | ~2 min |
| 3 | `03_chips.py` | Chips de campo (não-caju só onde é confiável) + até 30/30 chips por município | ~8 min |
| 4 | `02c_build_cache.py` | Cache memmap (float16) | ~6 min |
| 5 | `02d_cache_u8.py` | Cache em 8 bits (exato, metade do tamanho) | ~4 min |
| 6 | `08_spatial_cv.py` (`FT_INIT=../v10/models/ensemble_unet_conc`) | CV de 5 folds por **ajuste fino** a partir da v10, fold a fold | ~1 h |
| 7 | `10b_oof_threshold.py` | Limiar global fora do fold + **trava IoU ≥ 0,65** | < 1 min |
| 8 | `17_fusao.py` | Escolhe o peso da fusão U-Net × XGBoost no mapa fora do fold de Serra do Mel | ~1 min |
| 9 | `14_final_train.py 100 <limiar>` | Modelo final: **100 épocas × 5 sementes**, aquecimento de 5 épocas, EMA, Lovász, sorteio 3× dos municípios fracos | ~4 h |
| 10 | `15_predict_final.py` | Predição dos 25 municípios: 5 modelos, TTA ×2, fusão, pós-processamento de 1 ha | ~1 h 15 |
| 11 | `32_mosaico.py` | Raster único (fusão em tudo) | ~4 min |
| 12 | `16_comparacao.py` | Comparação v8 × v10 × v10b × MapBiomas × PAM | < 1 min |
| 13 | `33_unet_so.py` | Versão só U-Net (limiar 0,40) + comparação só U-Net × fusão × PAM | ~3 min |
| 14 | `34_mapa_final_c.py` | **Mapa final, opção (c)**: fusão só no sertão CE/RN; só U-Net no litoral CE e no PI/PE | ~3 min |

Variáveis de ambiente usadas: `GPU_AUG=1` (preparo dos dados na GPU; já exportada no `run_v10b.sh`), `FT_INIT` (pesos iniciais do ajuste fino) e `MB_RASTER` (raster MapBiomas).

### Como acompanhar
```bash
tail -f v10b/logs/run_v10b.log | grep -E "^===|CNN fold|seed .* ep 100"
```

---

## 6. Atalho: só gerar o mapa com modelos já treinados

Se você já tiver `v10b/models/final_seed*.pt`, `v10b/reports/fusao.json`, os VRTs dos embeddings (`v8/data/ee_exports/2025/embeddings/*.vrt`) e o XGBoost regional (`v10a/data/fontes_a/*_xr.tif`):
```bash
cd v10b
GPU_AUG=1 python 15_predict_final.py && python 32_mosaico.py && python 33_unet_so.py && python 34_mapa_final_c.py
```
Para **outros municípios**, inclua-os em `v8/01_build_expressions.py` (lista de regiões) e em `REG` de `v10/01_fontes.py` (região de cada município), exporte os embeddings (Etapa 2) e rode o atalho.

---

## 7. Resultados

### Qualidade (CV espacial fora do fold, Serra do Mel, mesmos chips de teste)
| Versão | IoU, rótulos originais | IoU, rótulos corrigidos* | Precisão | Recall |
|---|---|---|---|---|
| v10 | 0,661 | 0,773 | 0,890 | 0,855 |
| v10a-ft | 0,675 | 0,793 | 0,896 | 0,873 |
| **v10b** | 0,667 | **0,804** | **0,904** | **0,879** |

\*Rótulos corrigidos = não-caju só onde é confiável (os polígonos de campo são incompletos).

### Área 2025 × PAM (25 municípios)
| Mapa | Total (ha) | Correlação com a PAM | Razão 0,5–2,5 | Erro típico** |
|---|---|---|---|---|
| v10 | 237.084 | 0,86 | 17/25 | 0,33 |
| v10b só U-Net | 239.337 | 0,95 | 21/25 | 0,24 |
| v10b fusão em tudo | 332.104 | 0,82 | 19/25 | 0,33 |
| **v10b final (opção c)** | **248.513** | **0,95** | **21/25** | **0,18** |

\*\*Mediana de |log(mapa/PAM)|.

Tabela por município: [`resultados/comparacao_unet_x_fusao.md`](resultados/comparacao_unet_x_fusao.md) e [`resultados/area_caju_2025_final.csv`](resultados/area_caju_2025_final.csv).

### Arquivos em `resultados/`
| Arquivo | Conteúdo |
|---|---|
| `caju_2025_25mun_FINAL_classe.tif` | Mapa final: 0 / 1 / 255 (fora) |
| `caju_2025_25mun_FINAL_prob.tif` | Probabilidade 0–100. Os limiares diferem por região: 30 no sertão CE/RN (fusão), 40 no resto |
| `area_caju_2025_final.csv` | Área por município, método usado e razão com a PAM |

---

## 8. Limitações conhecidas
- **Icapuí (0,08 × PAM) e Cerro Corá (0,16 × PAM) seguem subestimados**. Fortim e Buíque estão em ~0,45. No litoral do CE há confusão com coqueiro.
- A PAM entrou como **prior de quantidade de rótulos** (v10a/v10b), então a comparação com a PAM é otimista. Uma avaliação de acurácia por amostragem estratificada ainda está pendente.
- Rótulos de campo só existem em Serra do Mel (RN). As demais regiões dependem de rótulos automáticos.
- MapBiomas a 30 m reamostrada para 10 m: erro de borda de até 1 pixel de 30 m.

---

## 9. Documentação detalhada
- [`docs/DECISOES.md`](docs/DECISOES.md): registro de todas as decisões (data, motivo, quem decidiu)
- [`docs/VERSOES_SINTESE.md`](docs/VERSOES_SINTESE.md): síntese das versões
- [`v7/V7_PILOTO_2025.md`](v7/V7_PILOTO_2025.md), [`v8/V8_PILOTO_25MUN_2025.md`](v8/V8_PILOTO_25MUN_2025.md), [`v10/V10_AMOSTRAS_CONCORDANCIA.md`](v10/V10_AMOSTRAS_CONCORDANCIA.md), [`v10a/V10A_AJUSTE_FINO.md`](v10a/V10A_AJUSTE_FINO.md), [`v10b/V10B_MAPBIOMAS_100EP.md`](v10b/V10B_MAPBIOMAS_100EP.md)

**Fontes de dados:** Google Satellite Embedding V1 (AlphaEarth Foundations); MapBiomas Coleção 11; IBGE PAM (SIDRA 1613) e malha municipal 2024.
