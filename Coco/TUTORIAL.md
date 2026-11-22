# Tutorial — como mapear coco com estes scripts

Do zero ao mapa: estatística (PAM/Censo) → embeddings → rótulos sem campo → treino U-Net → mapa e comparação.
Cada script imprime um resumo curto e é **retomável** (pula saídas que já existem). Os tempos são da v1 (RTX 3050 4 GB).

## 0. Pré-requisitos
| Item | Observação |
|---|---|
| Python 3.11 + GPU NVIDIA (CUDA) | `pip install -r requirements.txt` (torch na versão CUDA, ver pytorch.org) |
| Conta Google Earth Engine | só para exportar os embeddings (`scripts/run_export.sh`, ambiente com DatensEE) |
| Google Earth Pro | para digitalizar as amostras iniciais |
| Dados externos | malha municipal IBGE, MapBiomas (coleção com o ano do mapa), pesos do modelo de caju (inicialização) |
| Disco | ~35 GB por 15 municípios (embeddings 23 GB + cache de chips 7 GB) |

## 1. Configuração — `config/config.yaml`
Único lugar com caminhos e parâmetros. Ajuste:
- `data_root` (onde ficam dados e saídas; a versão, ex. `v1`, vira subpasta) e os caminhos em `externos`;
- `listas_municipios` (códigos IBGE) e `recorte.escopo` (lista usada);
- `rotulos.*` (classes MapBiomas compatíveis/seguras, erosão, mancha mínima), `treino.*`, `predicao.*`.

Todos os comandos abaixo rodam dentro de `scripts/` (ex.: `cd scripts`).

## 2. Estatística municipal (referência de área)
```bash
python 00_pam_coco.py          # PAM/IBGE tabela 1613, média 2023–25 (área destinada à colheita)
python 01_censo_agro_coco.py   # Censo Agropecuário 2017 (SIDRA 6955): área existente, nº de pés
```
Saída: `dados_externos/pam/pam_censo_coco.csv`. A área de referência por município é `max(PAM, Censo)`.

## 3. Embeddings AlphaEarth 2025 (64 bandas, 10 m)
```bash
python 03_regioes_export.py top15_caatinga   # recortes por município (UTM local, buffer 1 km)
bash run_export.sh top15_caatinga            # exporta os tiles (Earth Engine); ~1–2 h por município
python 09_recortes_fontes.py top15_caatinga  # MapBiomas, Descals e caju na grade de cada município
```

## 4. Rótulos sem dados de campo
**4a. Amostras-modelo (você).** No Google Earth Pro, crie a pasta `COCO` e digitalize 2+ polígonos de coqueiral
certo (copas em estrela, grade regular de ~8 m). Opcional: pasta `NAO_COCO`. Salve como KMZ em `amostras_usuario.kml_dir`.

**4b. Candidatos por similaridade** (onde houver amostras-modelo, ex. litoral):
```bash
python 05_candidatos_similaridade.py top15_caatinga   # kNN cosseno (k=5) aos pixels-modelo → KML de candidatos
```
**4c. Regiões sem amostras** (ex. São Francisco): agrupa manchas MapBiomas 48/41 e gera folhas de imagem de alta resolução
para identificar a copa; os grupos confirmados viram modelo regional.
```bash
python 06_grupos_perenes.py sao_francisco
python 07_folhas_visuais.py sao_francisco     # PNG em v1/data/rotulos/grupos/ — ver os critérios de copa no script
python 08_modelo_de_grupo.py sao_francisco 11,6   # grupos confirmados como coco
```
**4d. Conferência e rótulos finais**
```bash
python 10_qc_candidatos.py <tag> <gpkg_candidatos>   # folhas para conferir candidatos
python 12_importar_rotulos.py                        # amostras + conferências → r_ouro.gpkg
python 13_xgb_rotulos.py top15_caatinga              # XGBoost regional (GPU) → probabilidade por pixel
python 14_rotulos_regras.py top15_caatinga           # coco = maiores probabilidades até 0,6 × área de referência
python 15_qc_rotulos.py top15_caatinga               # QC cego: 18 pontos por folha
```
No QC, preencha `v1/data/rotulos/qc/qc_rotulos_decisoes.csv` (`id, classe_visual`) e rode o 12 de novo. **Critério de
aprovação:** ≥ 0,90 de acerto no coco e ≥ 0,95 no não-coco; se reprovar, revise as regras (`config` → `rotulos`) e refaça 13–15.

## 5. Treino
```bash
python 20_chips_cache.py top15_caatinga   # chips 256 px + cache uint8 (~7 GB)
python 30_treino.py cv 60 15              # validação cruzada espacial 5 folds (~2–4 h)
python 31_limiar_oof.py                   # limiar global + trava IoU ≥ 0,65 → v1/reports/cv_resumo.md
python 30_treino.py final 100 42          # modelo final; repita para as sementes 7, 123, 2024, 99 (~37 min cada)
```
A U-Net (resnet18, 64 canais) parte dos pesos do caju (`externos.caju_models`).

## 6. Mapa e produtos
```bash
python 40_predicao.py top15_caatinga      # ensemble 5 modelos + espelhamento; manchas ≥ 1 ha (~20 min)
python 41_mosaico.py top15_caatinga       # raster único EPSG:32724
python 42_vetores_kmz.py top15_caatinga   # polígonos KMZ/GPKG para Google Earth
python 50_comparacao.py top15_caatinga    # mapa × PAM × Censo × MapBiomas × Descals
```
**Tudo de uma vez (5 → 6):** `bash run_treino_mapa.sh` — espera a GPU liberar (`gpu_ocupada.ps1`), roda as etapas e
registra marcos em `v1/logs/run_treino_mapa.log`.

## 7. Como ler o resultado
- **IoU da validação cruzada** compara com os rótulos automáticos (passo 4); só os pontos conferidos medem a precisão real.
- **Razão mapa/PAM** por município: entre 0,5 e 1,3 é coerente; ≤ 0,1 indica falta de amostras daquele tipo de coqueiral.
- Na v1 o litoral do CE fora de Paraipaba ficou subestimado (coqueiral de sequeiro/disperso pouco amostrado).
  **Para melhorar:** digitalize amostras desse tipo (passo 4a) nesses municípios, refaça 05 → 15 e retreine (passo 5).

## 8. Adaptar para outra cultura ou região
1. Troque o produto em `pam`/`censo_agro` e a lista de municípios.
2. Digitalize amostras-modelo da nova cultura (pasta com o nome da cultura) e reescreva os critérios de copa do 07.
3. Ajuste `rotulos.classes_compativeis_mb` (classes MapBiomas onde a cultura pode estar).
4. Novo `project.version` (ex. `v2`) para não sobrescrever a anterior.

## 9. Problemas comuns
| Sintoma | Solução |
|---|---|
| Exportação falha com caminho `D:/...` | `run_export.sh` usa caminhos relativos (o Java lê `D:` como URL) |
| Erro de codificação no Windows | `export PYTHONUTF8=1 PYTHONIOENCODING=utf-8` |
| Falta de memória na GPU | outro processo usando a GPU: `nvidia-smi`; o encadeador espera liberar |
| Chips vazios | rótulos sem cobertura de embeddings — conferir os tiles do passo 3 |
