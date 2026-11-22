# Síntese das versões — Mapeamento de caju (fusão Sentinel-2 + Sentinel-1)

Documento-resumo das versões do experimento em `D:\Modelo_Caju`. Cada versão é
uma pasta autocontida (código + `config` + dados), compartilhando o mesmo
ambiente Python (`D:\Modelo_Caju\.venv`).

Local: `D:\Modelo_Caju\{v1, v2, v3, v4, v5}` · Ambiente: `.venv` na raiz (compartilhado).

---

## Visão geral (o que muda entre versões)

| Item | v1 | v2 | v3 | v4 | v5 |
|------|----|----|----|----|----|
| Amostras (polígonos) | 126 | 126 (do v1) | 466 | 466 | 466 |
| Origem dos COGs | download | reaproveita v1 | download | download | download |
| `chips.stride` | 128 | 64 | 64 | 64 | 64 |
| Índices espectrais | 6 | 6 | 12 | 6 (enxuto) | 6 (enxuto) |
| Textura SAR | não | não | **sim** | sim | sim |
| Canais de entrada | ~44 | ~44 | 60 | ~52 | ~52 |
| `training.loss` | bce_dice | focal_dice | focal_dice | focal_dice | focal_dice |
| `split.block_size_km` | 20 | 20 | 20 | **10** | 10 |
| `split.fractions` | 0.6/0.2/0.2* | 0.6/0.2/0.2 | 0.6/0.2/0.2 | 0.6/0.2/0.2 | 0.6/0.2/0.2 |
| `model.encoder` | resnet34 | resnet34 | resnet34 | resnet18 | **mobilenet_v2** |
| `training.weight_decay` | 1e-4 | 1e-4 | 1e-4 | **1e-3** | 1e-3 |
| `inference.threshold_criterion` | f1 | f1 | **min_precision** | min_precision | min_precision |

\* No v1 o `fractions` começou em 0.70/0.15/0.15 e foi ajustado para 0.60/0.20/0.20
porque, com poucos blocos espaciais, o conjunto de teste ficava vazio.

---

## v1 — linha de base

**Objetivo:** primeira execução funcional do pipeline (E3, fusão S2+S1).

**Características:**
- 126 polígonos de treino; download dos compostos multissensor (COGs de features).
- Chips com `stride 128` (sobreposição de 50%): 369 chips.
- 6 índices espectrais: NDVI, NDRE, NDMI, EVI, SAVI, NBR.
- Perda `bce_dice`; encoder ResNet-34; split por blocos de 20 km.

**Correções aplicadas durante o v1 (valem para todas as versões seguintes):**
- `output_root` apontado para a pasta de trabalho.
- **NumPy fixado < 2** e **`fill_value=np.float32(nan)`** no stackstac (S2 e S1) —
  compatibilidade com `stackstac 0.5.1`.
- **`select_valid_scenes`**: indexação por posição (evita conflito da coord `band`).
- **`compute_indices`** e **`build_feature_stack`**: limpeza de coordenadas antes do
  `xr.concat` (evita "coordinate ... not present in all datasets").
- **`split.fractions`** 0.60/0.20/0.20 (teste não-vazio com poucos blocos).

**Resultado observado:** `best` de validação (IoU do caju) ~ **0,31**.

---

## v2 — mais amostras a partir dos mesmos downloads

**Objetivo:** aumentar o número de chips sem baixar de novo e atacar o
desbalanceamento.

**Mudanças em relação ao v1:**
- `chips.stride: 64` (sobreposição ~75%) → muitos mais chips por área.
- `training.loss: focal_dice` → foca a classe rara (caju).
- Reaproveita os COGs do v1 (`paths.processed` aponta para os features do v1).

**Resultado observado (teste):** recall alto (~0,79) e **precisão baixa (~0,25)**;
IoU de teste ~0,235. Acha quase todo o caju, mas superestima a área.

---

## v3 — nova base de campo + mais índices + foco em precisão + textura SAR

**Objetivo:** usar a base ampliada, melhorar a separação espectral e reduzir
falsos positivos.

**Mudanças em relação ao v2:**
- **466 polígonos** (base de campo ampliada); download próprio.
- **12 índices espectrais** (6 novos): + NDRE2, CIre, GNDVI, NDWI, MSI, RVI
  (red-edge/clorofila e umidade). Adicionada a função `_ratio`.
- `inference.threshold_criterion: min_precision` (min 0,80).
- **Textura SAR** adicionada: estatísticas locais (std, range) + GLCM (contrast,
  homogeneity, dissimilarity) sobre a mediana temporal de VV/VH → **60 canais**.
- Scripts `01/02/04` resolvem o vetor a partir de `paths.vectors`.
- **`validate_geometries`** corrigido: remove geometrias nulas/vazias **antes** do
  `make_valid` (o shapefile novo tinha feições sem geometria).
- Reconstrução do `Pol_Caju.shp` (estava vazio) a partir da base real (466 polígonos).

**Resultado observado (treino):** overfitting — IoU treino ~0,78 vs val ~0,31;
`val_loss` acima do `train_loss`. Causa: só **5 blocos espaciais** de 20 km + alta
dimensionalidade (60 canais). Validação de 1 bloco é ruidosa.

---

## v4 — anti-overfitting

**Objetivo:** reduzir o overfitting e tornar a métrica mais confiável.

**Mudanças em relação ao v3:**
- **Regularização:** `model.encoder: resnet18` e `training.weight_decay: 1e-3`.
- **`split.block_size_km: 10`** → mais blocos → val/teste mais confiáveis.
- **Índices enxutos (6):** NDVI, NDRE, NDMI, EVI, NBR, NDWI (remove RVI/SAVI/
  NDRE2/CIre/GNDVI/MSI, redundantes).
- Mantém: `stride 64`, `focal_dice`, `min_precision`, textura SAR.
- Download próprio (os canais mudaram).

---

## v5 — encoder leve (mobilenet_v2)

**Objetivo:** empurrar ainda mais a regularização com um encoder de menos
parâmetros, mantendo o resto igual ao v4 (comparação limpa de encoder).

**Mudanças em relação ao v4:**
- **`model.encoder: mobilenet_v2`** (~2,2 M de parâmetros, contra ~11 M do
  resnet18) → menor capacidade → tende a overfittar menos com área concentrada.
- Todo o restante idêntico ao v4: blocos de 10 km, 6 índices, `focal_dice`,
  `min_precision`, `weight_decay 1e-3`, `stride 64`, textura SAR.
- `encoder_weights: imagenet` (warm start nos 3 primeiros canais; o smp adapta a
  conv de entrada para os ~52 canais). Se quiser, testar `null` (treino do zero).

**Como rodar (a partir de `D:\Modelo_Caju\v5`):**
```powershell
D:\Modelo_Caju\.venv\Scripts\Activate.ps1
cd D:\Modelo_Caju\v5
python 01_download_training_data.py --vector Pol_Caju.shp --role ground_truth
python 02_create_training_chips.py  --vector Pol_Caju.shp
python 03_train_model.py
```

Observação: v4 (resnet18) e v5 (mobilenet_v2) diferem **apenas no encoder** — os
dados/chips são gerados igual, então comparar o IoU/precisão/recall dos dois é um
teste direto de qual encoder generaliza melhor nesta área.

---

## Textura SAR (v3+)

Calculada sobre a mediana temporal de VV/VH (dB), 10 canais (5 por polarização):
- **Estatísticas locais** (janela 5×5): `std`, `range` (rugosidade/heterogeneidade).
- **GLCM/Haralick** (diferenças deslocadas + média local, 4 orientações):
  `contrast`, `homogeneity`, `dissimilarity`.

Motivo: o dossel estruturado do cajueiro gera um padrão espacial de
retroespalhamento diferente de pasto/solo/água (lisos), ajudando a precisão.

---

## Resultados e comparação (teste, classe caju)

Métricas do conjunto de TESTE lidas dos `reports\metrics_E3_fusion.json` de cada
versão. **Atenção à comparabilidade** (ver ressalvas abaixo).

| Versão | Encoder | Índices/canais | Bloco split | n_test (chips) | Limiar | Precisão | Recall | F1 | IoU (CNN) | IoU (RF) |
|---|---|---|---|---|---|---|---|---|---|---|
| v3 | resnet34 | 12 / 60 | 20 km | 576 | 0,50 | 0,794 | 0,867 | 0,829 | **0,708** | 0,600 |
| **v4** | resnet18 | 6 / 54 | 10 km | 490 | 0,90 | 0,809 | 0,741 | 0,774 | **0,631** | 0,488 |
| v5 | mobilenet_v2 | 6 / 54 | 10 km | 490 | 0,85 | 0,818 | 0,632 | 0,713 | **0,554** | 0,488 |

Melhor IoU de validação (`best`, monitorado a t=0,5): v4 = 0,612 · v5 = 0,602 (empate técnico).

### Ressalvas de comparabilidade (importantes)

- **v4 × v5 SÃO comparáveis:** mesmo split de 10 km, mesmos dados, mesma semente
  (`n_test = 490` idêntico e **RF baseline idêntico, IoU 0,488**, nos dois — o RF é
  determinístico no mesmo split). Só muda o encoder. → **O v4 (resnet18) vence:**
  IoU 0,631 vs 0,554, F1 0,774 vs 0,713, recall 0,741 vs 0,632, precisão ~igual.
  Apesar de o v4 overfittar mais no treino, generalizou melhor no teste.
- **v3 NÃO é comparável direto** com v4/v5: usou blocos de **20 km** (teste é 1
  bloco grande, menos independente → otimista), **12 índices** (60 canais) e
  **limiar 0,50** (equilibrado). O IoU 0,708 parece o melhor, mas é inflado pela
  granularidade do split. Prova: o **RF baseline** cai de **0,600 (20 km)** para
  **0,488 (10 km)** — o modelo não piorou; o split de 10 km é mais honesto/difícil.

### Conclusões

- **Deep Learning > baseline tradicional** no split honesto: v4 CNN 0,631 vs RF
  0,488 → **+0,143 de IoU**. Argumento forte para a questão científica (fusão +
  CNN supera o método clássico por pixel).
- **Melhor modelo com avaliação confiável: v4 (resnet18)** — IoU ~0,63, F1 ~0,77,
  precisão 0,81, recall 0,74 (10 km).
- O ponto de operação (limiar) é ajustável **sem re-treinar**: `min_precision`
  levou v4/v5 a limiares altos (0,85–0,90), priorizando precisão ao custo de
  recall. Para mapa de área, um limiar menor recupera recall.
- **Ainda é 1 split.** Para cravar "v4 > v5" e dar o número definitivo com barra de
  incerteza, rodar a **validação cruzada espacial** (`08_spatial_cv.py`).

### Nota de execução (tempo)

Treinos rodaram em **CPU** (~400–700 s/época), então cada versão levou horas.
Sequência típica do `03`: loop de épocas → early stopping → TESTE → **baseline RF
(passo lento, ~45 min em CPU, "silencioso")** → grava o relatório. Ex. v4: early
stopping na época 66 (14:34) → TESTE (14:37) → relatório salvo (15:24). O RF é o
trecho que parece "travado"; use `--skip-baseline` para pulá-lo. Migrar para GPU
(reinstalar PyTorch CUDA) reduz drasticamente esses tempos.

---

## Limitação de fundo (todas as versões)

O fator que mais limita o desempenho é a **cobertura espacial dos rótulos**: a
área de campo é pequena e concentrada (poucos blocos). Adensar amostras no mesmo
local aumenta a contagem de chips, mas não ensina o modelo a generalizar para
paisagens novas. Para furar o teto de ~0,31 de IoU de validação de forma
consistente, o caminho principal é **rotular caju em mais regiões distintas**.

Alternativas de modelagem discutidas (para escassez/concentração de rótulos):
gradient boosting por pixel (XGBoost/LightGBM), modelos de fundação geoespaciais
(Prithvi, Clay, SatMAE, Presto), OBIA + classificador, semi-supervisão
(pseudo-rótulos) e aprendizado PU. Truques: validação cruzada espacial
(leave-one-block-out), loss Tversky/Focal-Tversky, seleção de features (top-k por
importância), SWA, ensemble e hard-negative mining.

---

## Saídas de cada versão (padrão)

Dentro de cada `vX\` (resolvidas sob o `output_root`):
- `data\processed\features_*.tif` — COGs de features por cluster (treino).
- `data\chips\*.npz` + `metadata\chips_metadata.parquet` — chips e metadados.
- `models\E3_fusion_best.pt` — melhor checkpoint (por IoU de validação).
- `metadata\norm_stats.json` — estatísticas de normalização (só do treino).
- `reports\metrics_E3_fusion.json` — métricas de teste + threshold + baseline RF.

---

## v6 — GPU + dedup S2 + multi-ano + pseudo-rótulos + ensemble (nov/2026)

Base: v4. Detalhes completos em `v6\V6_MUDANCAS.md`.
- Treino na **GPU** (PyTorch `2.14.1+cu130` no `.venv`; antes era `+cpu`, ~10 h por treino).
- **Dedup de cenas S2** (original + reprocessada): 113 → 62 cenas por cluster.
- **Multi-ano** 2022/2023/2024 com os mesmos polígonos (`--year` no 01).
- **Pseudo-rótulos** dos mapas v4 (p ≥ 0,95 / ≤ 0,02), só no treino, longe de val/teste (`02b_pseudo_label_chips.py`).
- **Ensemble** dos 5 folds do CV espacial (`08` salva os folds; `05 --ensemble`).
- Chips float16 sem compressão; stride 96; limites de RAM (dask 4 threads, 2 workers).
