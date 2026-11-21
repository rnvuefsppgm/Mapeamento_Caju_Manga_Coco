# v7 — Piloto RN 2025: U-Net com embeddings × Sentinel × Sentinel+embeddings

Data: 20/11/2026 · Base: código da **v6** · Pasta: `D:\Modelo_Caju\v7` · Ambiente: `.venv` (PyTorch `2.14.1+cu130`, GPU RTX 3050).

## Objetivo
Comparar três U-Nets (resnet18) **nas mesmas dobras do CV espacial (5 folds, blocos de 10 km), com os mesmos chips e as mesmas amostras**. O que muda entre elas são só os canais de entrada:

| Experimento | Canais | Entrada |
|---|---|---|
| `unet_emb` | 64 | Google Satellite Embedding (AlphaEarth) 2025 |
| `unet_sentinel` | 54 | Compostos S2 + S1 (mesmo esquema da v4/v6) maio–out/2025 |
| `unet_sentinel_emb` | 118 | Os dois juntos |

Área: **só a área de treino**, os 3 clusters em torno dos 466 polígonos (lon −37,17 a −36,87; lat −5,32 a −4,99; principalmente Serra do Mel). Ano: 2025.

> Replanejado em 20/11/2026 às 14:05: o 04 (6 municípios inteiros) foi cancelado; pseudo-rótulos e mapa final ficam sobre os clusters de treino. A v6 (multi-ano) foi interrompida durante 2023; os compostos de 2022 ficaram salvos em `v6/data/processed`.

## Mudanças em relação à v6

| Mudança | Arquivo(s) |
|---|---|
| Janela temporal de 2025 (01 e 04 geram compostos de 2025) | `config/config.yaml` (`temporal`) |
| **Download de embeddings sem tarefas no Earth Engine:** `computePixels` síncrono em tiles de 512 px, 6 em paralelo, na **mesma grade** de cada COG Sentinel (CRS, transform e dimensões). Armazenados em int8 (valor × 127; −128 = NoData). Retomável | **novo** `src/embeddings.py`, **novo** `03a_download_embeddings.py`, config `embeddings` |
| Chips guardam `[Sentinel ; embeddings]` (118 canais, float16); o modelo escolhe os canais por `experiment.channels` | `02_create_training_chips.py`, `src/dataset.py` (`select_channels`) |
| `08_spatial_cv.py --channels {sentinel,emb,both} --name X`: mesmos folds para os três, tempo por fold registrado e ensemble salvo em `models/ensemble_<X>/` | `08_spatial_cv.py` |
| Pseudo-rótulos: chips extras (stride 256) sobre os **clusters de treino 2025**, com a probabilidade v4/2023 de Serra do Mel e Porto do Mangue reprojetada e mosaicada na grade de cada cluster | `02b_pseudo_label_chips.py` |
| **Pontos para revisão manual:** 150 pontos por município nas zonas incertas (0,3 ≤ p ≤ 0,7), com o campo `classe_revisada` para preencher no QGIS | `02b_pseudo_label_chips.py` → `data/labels/review_points.gpkg` |
| Comparação automática (média ± dp, IoU por fold, quantos folds cada modelo vence) | **novo** `10_compare_models.py` → `reports/comparacao_v7.md` |
| Predição com ensemble do melhor modelo, juntando os embeddings quando necessário | `05_predict.py --ensemble <nome>` |

Melhorias de amostra herdadas da v6 e aplicadas aos três modelos: deduplicação de cenas S2; pseudo-rótulos (p ≥ 0,95 / ≤ 0,02) só no treino, longe de val/teste e dos blocos vizinhos; chips float16.

## Execução

```bash
cd D:/Modelo_Caju/v7
./run_v7_data.sh                    # fase A: Sentinel 2025 (01 treino + 04 municípios)
./run_v7_train.sh rnv-geodatin      # fase B: embeddings -> 02 -> 02b -> 08 x3 (GPU) -> 10 -> 05
```
A fase B espera a fase A terminar e só começa o treino depois que a v6 liberar a GPU. Log: `logs/run_v7.log`.

Requisito: um projeto Google Cloud **registrado no Earth Engine** (credenciais já existem em `~/.config/earthengine`).

## Estimativa de tempo (antes de iniciar)

| Etapa | Estimativa | Real |
|---|---|---|
| Montar v7, instalar `earthengine-api` e `tabulate`, testar GPU | 20–30 min | ~30 min |
| 01: compostos de treino 2025 (3 clusters) | 40–60 min | |
| ~~04: compostos dos 6 municípios~~ | cancelado | — |
| 03a: embeddings 2025 (int8, ~5 GB) | 30–90 min | |
| 02 + 02b: chips, pseudo-rótulos e pontos de revisão | 20–30 min | |
| 08 × 3: CV de 5 folds na GPU (GPU livre) | 5–7 h | |
| 10 + 05: comparação e mapa da área de treino | ~20 min | |

## Testes feitos
- `py_compile` em todos os scripts: OK.
- `select_channels`: 118 canais → sentinel 54 / emb 64 / both 118: OK.
- GPU (na v6): batch 16 → 1,4 GB de VRAM, 0,15 s/passo.
- Earth Engine: projeto `rnv-geodatin` OK. Embeddings disponíveis de 2017 a **2025**. Teste com `computePixels` (64×64 px, EPSG:32724, int8): 64 bandas OK, sem tarefas.

## Resultados
_Preencher a partir de `reports/comparacao_v7.md`._

## Limitações
- Os pseudo-rótulos vêm do modelo v4/2023 aplicado à paisagem de 2025: plantios novos e cortes entre 2023 e 2025 viram ruído.
- As amostras de campo continuam 91% em Serra do Mel. Para generalizar, revise `review_points.gpkg` e colete polígonos nos outros municípios.
- Se a camada 2025 dos embeddings não estiver publicada, o 03a para com erro; nesse caso use `embeddings.year: 2024` e registre a diferença de ano.

## Status (20/11/2026)

- `unet_emb`: 5 folds concluídos, IoU 0,563 ± 0,156 (folds 0,710 / 0,725 / 0,602 / 0,468 / 0,311).
- `unet_sentinel` (interrompido no fold 1) e `unet_sentinel_emb` (cancelado): o usuário decidiu seguir **só com embeddings**. Continuação em `v8` (predição) e `v9` (retreino).
