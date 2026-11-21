# v8 — Piloto 25 municípios, 2025, só embeddings (fluxo DatensEE)

Data: 20/11/2026 · Pasta: `D:\Modelo_Caju\v8` · Fluxo com Sentinel do Earth Engine guardado (sem executar) em `D:\Modelo_Caju\v8_sentinel`.

## Objetivo
Mapa de caju 2025 para os **25 municípios da Caatinga com maior produção de castanha** (PAM/IBGE, média 2023–2025; 73,7% da produção da Caatinga). O modelo é o ensemble **`unet_emb` da v7**, treinado só em Serra do Mel.

## Fluxo
| # | Script | Ambiente | O que faz |
|---|---|---|---|
| 1 | `01_build_expressions.py` | `.venv` | Expressão EE dos embeddings 2025 (uint8 = round(x·127)+128; 0 = NoData) e regiões GeoJSON (25 municípios com buffer de 1 km + 3 clusters de treino) |
| 2 | `run_v8_export.sh` → `export_one.sh` | `.venv_v8` (Python 3.14 + DatensEE 0.1.0a8 + JAR) | `datensee export` no modo local (High Volume API, **sem tarefas EE**): tiles de 512 px, EPSG:32724, 10 m, `retry` até 3×. Retomável |
| 3 | `03_mosaic_vrt.py` | `.venv` | Um VRT por região apontando para os tiles (sem duplicar dados) |
| 4 | `04_predict_emb.py` | `.venv` (GPU) | Predição em blocos de 2048 px (+128 de margem) com o ensemble da v7; recorte ao polígono; saídas `predictions/2025/<regiao>_prob.tif` (0–100) e `_class.tif` |
| 5 | `05_compare_pam.py` | `.venv` | Área mapeada × área colhida PAM → `reports/comparacao_pam_2025.md` |

`run_v8_predict.sh` encadeia 3→5 e espera as exportações terminarem e o `fold5.pt` do `unet_emb` existir.

## Pastas
```
v8/config, expressions/        expressões EE serializadas (JSON) + nomes das bandas
v8/data/aoi/regions/           28 GeoJSON (Polygon puro, exigência do DatensEE) + regions.txt
v8/data/ee_exports/2025/embeddings/<regiao>/  tiles COG + _export_meta.json; <regiao>.vrt
v8/predictions/2025/           mapas por município
v8/reports/, v8/logs/          area_caju_2025.json, comparacao_pam_2025.md, export.log, predict.log
dados_externos/                PAM 1613, biomas IBGE, ranking, municipios25.gpkg (de D:\arc_arquivos\BR_Municipios_2024)
```

## Decisões e ajustes
- **Credencial:** o DatensEE usa a credencial padrão do Google Cloud. Ela foi criada a partir do token do Earth Engine (mesmo `refresh_token`, projeto `rnv-geodatin`) em `%APPDATA%\gcloud\application_default_credentials.json`, com autorização do usuário.
- **Ambiente separado** `.venv_v8` (o DatensEE exige Python ≥ 3.12; o `.venv` do treino é 3.11 e não foi tocado).
- **Região:** o DatensEE recusa FeatureCollection, então as regiões são gravadas como Polygon puro.
- **Tipo de dado:** o DatensEE não aceita int8, então os embeddings vão em **uint8** com deslocamento de +128. Decodificados, são idênticos aos da v7.

## Testes
| Teste | Resultado |
|---|---|
| Exportar `treino_blk_18_236` | 9 tiles, 36 s, 88 MB, 0 falhas |
| Tile do DatensEE × embeddings da v7 (mesma área) | **100% dos 16,8 M valores iguais**; grade alinhada (deslocamento inteiro de 104/200 px) |
| Exportar `treino_blk_17_235` e `treino_blk_18_235` | 24 tiles cada, ~40 s, ~240 MB |
| **Exportação completa (28 regiões)** | 16:59 → 17:31, **15 GB**, 0 falhas; 31–108 s por município (Granja: 153 tiles, 1,6 GB). Bug: a última linha de `regions.txt` (Monsenhor Hipólito) era ignorada por falta de quebra de linha; corrigido em `run_v8_export.sh` e exportada à parte (43 s) |
| Predição ponta a ponta em `treino_blk_18_236` (2 folds) | ~1 min; 2.526 ha de caju; limiar médio 0,35 |

## Estimativa de espaço (só embeddings)
~10 GB (realista) a ~16 GB (máximo) para os 25 municípios, mais ~1 GB de mapas. O Sentinel do Earth Engine (~22–37 GB) ficou fora da v8.

## Resultados
_Preencher a partir de `reports/comparacao_pam_2025.md`._

## Limitações
- Treino só em Serra do Mel: nos outros 24 municípios **não há validação**. Use a comparação com a PAM como sinal e revise visualmente as amostras.
- O mapa mede área plantada; a PAM mede área colhida declarada.
