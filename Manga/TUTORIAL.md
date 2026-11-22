# Tutorial — como usar os scripts do mapeamento de manga

Passo a passo para reproduzir o mapa de manga 2025 (ou adaptar para outra cultura ou região). Cada etapa grava um log em `logs/` e um resumo em `reports/`. As cadeias (`run_*.sh`) são **retomáveis**: se pararem, rode de novo e elas pulam o que já foi feito.

---

## 0. Pré-requisitos

| Item | Detalhe |
|---|---|
| Sistema | Windows 10/11 com Git Bash (os `.sh` rodam no Bash; os `.py` em qualquer terminal) |
| GPU | NVIDIA com CUDA (testado numa RTX 3050 de 4 GB). Sem GPU funciona, mas o treino fica muito lento |
| Python principal | 3.11 com `torch` (CUDA), `segmentation_models_pytorch`, `xgboost` ≥ 3 (GPU), `cupy-cuda13x`, `geopandas`, `rasterio`, `scipy`, `scikit-learn`, `pandas`, `pyarrow`, `pyyaml`, `requests`, `tabulate`, `tqdm` |
| Python da exportação | ≥ 3.12 com **DatensEE** (exportação local dos embeddings pela High Volume API do Earth Engine, sem tarefas) |
| Earth Engine | projeto Google Cloud registrado no Earth Engine e credencial padrão (`gcloud auth application-default login` ou equivalente) |
| Dados externos | malha municipal IBGE (`BR_Municipios_2024.shp`), CSV de bioma predominante por município, MapBiomas Coleção 11 2025 (GeoTIFF Brasil, 30 m) |
| Opcional | pesos finais de uma U-Net já treinada com os mesmos 64 embeddings (aqui, a do caju v10b) para a transferência de aprendizado |

Exemplo de ambiente:
```bash
python -m venv .venv && source .venv/Scripts/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install segmentation-models-pytorch "xgboost>=3" cupy-cuda13x geopandas rasterio scipy scikit-learn pandas pyarrow pyyaml requests tabulate tqdm
```

---

## 1. Configuração (`config/config.yaml`)

**Todos os caminhos e parâmetros ficam no config.** Antes de rodar, ajuste:

| Chave | O que é |
|---|---|
| `output_root` | pasta do projeto (onde ficam `dados/`, `modelos/`, `reports/`) |
| `manga.fontes.*` | malha municipal, CSV de bioma, GeoTIFF da MapBiomas, ambiente do DatensEE |
| `manga.pam` | tabela SIDRA 1613 e código do produto (Manga = 2737; Coco-da-baía = 2727; Castanha de caju = 2725) |
| `manga.selecao` | bioma, número de municípios e códigos IBGE que entram sempre (o município das amostras) |
| `manga.censo_agro` | tabela SIDRA 6955, produto da classificação 227 (Manga = 4968) |
| `manga.embeddings.projeto_ee` | **seu** projeto do Earth Engine |
| `amostragem.*` | parâmetros dos candidatos e da classificação automática |
| `treino_v1.*` | municípios de treino, cotas de chips, pesos iniciais, folds, épocas, sementes, trava de IoU |

Para ver o estado do projeto a qualquer momento:
```bash
python 00_status.py
```

---

## 2. Entrada: suas amostras

1. No Google Earth Pro, crie uma pasta **`MANGA`** e digitalize polígonos de pomares que você tem certeza de que são manga (talhão inteiro, sem estrada nem casa).
2. Salve como KML ou KMZ (botão direito na pasta → *Salvar lugar como*) em `entrada/`.
3. Dica: quanto mais regiões diferentes, melhor. Com amostras de uma só paisagem, o mapa só fica confiável nessa paisagem (ver [Limitações no README](README.md#limitações)).

---

## 3. Etapa de dados (sem treino)

```bash
python 01_amostras.py          # KML/KMZ -> dados/amostras/manga_poligonos.gpkg (remove duplicatas)
python 02_pam_municipios.py    # baixa a PAM (SIDRA) e escolhe os N maiores municípios do bioma
python 03_regioes.py           # GeoJSON de cada município + 1 km para a exportação
bash   04_export_embeddings.sh m2404705_Ipanguacu   # TESTE com um município pequeno (~1 min)
bash   run_dados.sh            # exporta todos -> VRT (05) -> MapBiomas (06) -> rótulos (07) -> chips (08) -> cache (09)
```

| Saída | Onde |
|---|---|
| Ranking PAM e municípios escolhidos | `reports/02_pam_municipios.md`, `dados/pam/` |
| Embeddings 2025 (uint8, 64 bandas, NoData 0) | `dados/embeddings/<regiao>/` + `<regiao>.vrt` |
| Classes MapBiomas dentro das amostras | `reports/mb_classes_manga.csv` (define as classes compatíveis) |

> O DatensEE exige **caminhos relativos** (o Java lê `D:/...` como URL); por isso o `04` faz `cd` na pasta do projeto.

---

## 4. Etapa de amostras (método do coco, opcional mas recomendado)

```bash
python scripts_amostras/A1_censo_agro_manga.py                 # Censo 2017 x PAM x bioma -> reports/A1_ranking_manga.md
python scripts_amostras/A3_candidatos_similaridade.py top15_caatinga   # candidatos -> KML para o Google Earth
python scripts_amostras/A4a_classificacao_auto.py top15_caatinga       # MANGA_ok / NAO_MANGA / DUVIDA automático
python scripts_amostras/A5_mapa_xgb.py top15_caatinga                  # mapa automático por pixel (XGBoost)
python scripts_amostras/A5_tabela.py                                    # tabela parcial x PAM x Censo
```

- A **lista de municípios** (`top15_caatinga`) fica em `amostragem.listas_municipios` e em `dados/aoi/regions_<lista>.txt`.
- A3 gera `dados/rotulos/candidatos/candidatos_manga_<lista>.kml`. Para **conferir à mão**, mova cada candidato para as pastas `MANGA_ok`, `NAO_MANGA` ou `DUVIDA` e salve o KMZ em `dados/amostras_usuario/`.
- A4a faz essa separação **automaticamente** por probabilidade. Esses rótulos servem só para treino, **nunca para avaliação**. Detalhe em [`docs/A4a_CLASSIFICACAO_AUTO.md`](docs/A4a_CLASSIFICACAO_AUTO.md).
- A3 e A5 são retomáveis por município (cache em `dados/rotulos/candidatos/_cache_*` e `dados/predicoes/xgb_auto/`).

---

## 5. Treino da rede

```bash
bash run_treino_v1.sh
```
A cadeia faz:

| # | Script | O que faz |
|---|---|---|
| 10 | `10_rotulos_chapada.py` | rótulos de treino dos municípios de `treino_v1.municipios` (manga, não-manga, ignorado) |
| 08 | `08_chips.py` | chips de campo de 256 px (stride em `manga.chips_campo.stride`) |
| 11 | `11_chips_v1.py` | chips de campo (avaliação) + automáticos (só treino) + chips de caju como não-manga + cache 8 bits |
| 12 | `12_init_pesos.py` | pesos iniciais: **A** = modelo do caju com a cabeça reiniciada; **B** = ImageNet |
| 13 | `13_cv.py` | validação cruzada espacial (5 folds, blocos de 3 km) para A e para B, avaliada só nos rótulos conferidos |
| 14 | `14_limiar_oof.py` | limiar global fora do fold e IoU agregado -> `reports/oof_resumo.md` |
| – | trava | segue só se o melhor IoU ≥ `treino_v1.trava_iou` (0,60) |
| 15 | `15_final.py` | modelo final: 5 sementes × 100 épocas partindo do vencedor -> `modelos/final_seed*.pt` |

Variáveis úteis:
```bash
export GPU_AUG=1        # preparo dos lotes na GPU (bem mais rápido; exige o cache em 8 bits)
export GPU_FRAC=0.45    # fração máxima da memória da GPU (use se dividir a GPU com outro treino)
```

---

## 6. Predição e produtos

```bash
bash run_pos_treino_v1.sh     # 16 predição -> 17 mosaico -> 18 comparação -> 19 vetores
# ou, passo a passo:
python 16_predicao.py 0.55    # limiar do CV (reports/oof_v1_A.json)
python 17_mosaico.py
python 18_comparacao.py
python 19_vetores_kmz.py
```

| Produto | Onde |
|---|---|
| Raster por município (classe e probabilidade) | `dados/predicoes/v1/<código>_<nome>_{class,prob}.tif` |
| Raster único dos 25 municípios | `dados/predicoes/manga_2025_25mun_{classe,prob}.tif` (GeoTIFF esparso, EPSG:32724, 10 m) |
| Vetores para o Google Earth / GIS | `dados/predicoes/v1/vetores/*.kmz`, `*.gpkg` (área em ha e probabilidade média por mancha) |
| Comparação com PAM e Censo | `reports/v1_comparacao.md` |

Neste repositório, os mapas de **classe** e os **vetores** da v1 estão em [`resultados/`](resultados/). Os rasters de probabilidade (~100 MB) não foram incluídos por causa do tamanho.

---

## 7. Adaptar para outra cultura ou região

1. Troque o produto na PAM (`manga.pam.categoria_manga`) e no Censo (`manga.censo_agro.produto`). Os códigos estão nos metadados das tabelas 1613 e 6955 do SIDRA.
2. Troque `selecao.bioma`, `selecao.n_municipios` e `selecao.incluir_sempre` (municípios das suas amostras).
3. Recalcule as classes compatíveis da MapBiomas com `python 06_mapbiomas.py --classes` e atualize `manga.rotulos.classes_mb_compativeis` com as classes que somam cerca de 98% das amostras.
4. Ajuste `treino_v1.municipios` para os municípios com amostras confiáveis.
5. Sem pesos de outra cultura? Use só a largada B (ImageNet) ou aponte `treino_v1.init_caju` para outro modelo com os mesmos 64 canais.

---

## 8. Problemas comuns

| Sintoma | Causa e solução |
|---|---|
| `CUDA out of memory` na predição | outro processo na GPU ou `GPU_FRAC` baixo demais: rode com `GPU_FRAC=0.9` e a GPU livre |
| Treino 10× mais lento | dois treinos na mesma GPU de 4 GB estouram a memória e o Windows passa a usar a RAM: rode um de cada vez |
| `Os polígonos não caem em nenhuma região exportada` (A3) | a lista precisa incluir o município das amostras |
| Exportação do DatensEE falha com caminho `D:/...` | use caminhos relativos (o `04` já faz isso) |
| `UnicodeEncodeError` no terminal | `export PYTHONIOENCODING=utf-8` |
| Mapa muito acima da PAM numa região | confusão com outras frutíferas irrigadas: digitalize amostras de manga **e** de não-manga nessa região e treine de novo |
