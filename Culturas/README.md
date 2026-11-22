# Culturas — mapa único caju + manga + coco 2025

Junta os mapas finais de caju, manga e coco num raster único sobre o **MapBiomas Coleção 11 (2025)**, com códigos novos
que não existem na legenda MapBiomas: **97 caju · 98 manga · 99 coco**. 10 m, EPSG:32724, 58 municípios.

| Cultura (código) | Mapa (ha) | PAM 2023–25 (ha) | Censo 2017 (ha) |
|---|---|---|---|
| Caju (97), 25 municípios | 248.415 | 293.361 | 160.939 |
| Manga (98), 25 municípios | 26.395 | 56.473 | 43.015 |
| Coco (99), 15 municípios | 23.053 | 54.136 | 33.325 |

Sobreposição entre culturas: 458 ha (caju×coco 258, manga×coco 200), decidida pela maior margem normalizada
`(p − limiar)/(1 − limiar)` de cada modelo.

## Arquivos
| Caminho | Conteúdo |
|---|---|
| `saidas/mapa/culturas_2025_mapbiomas_c11.tif` | mapa único (paleta MapBiomas + 3 culturas embutida; 255 = fora) |
| `saidas/auxiliar/disputas_2025.tif` | pixels disputados (bits: 1 caju, 2 manga, 4 coco) |
| `saidas/tabelas/tabela_final_municipios.csv` | área mapeada × PAM × Censo por município e cultura |
| `saidas/tabelas/areas_municipio_classe.csv` | área de todas as classes (MapBiomas + culturas) por município |
| `saidas/tabelas/cobertura_municipios.csv` | quais culturas foram mapeadas em cada município |
| `dados/pam_censo/pam_censo_3culturas.csv` | PAM (IBGE 1613) e Censo Agro 2017 (IBGE 6955) das 3 culturas |
| `docs/CLASSES_MAPA_UNICO.md` | valores das classes, regras e tabela por município |
| `docs/DECISOES_METODOLOGICAS.md` | as 10 decisões tomadas, com motivo e efeito |
| `docs/PLANO.md` | plano da tarefa |

## Como refazer
Entradas: os mosaicos finais de cada cultura (pastas `../Caju/resultados`, `../Manga/resultados`, `../Coco/resultados`
ou as saídas locais dos projetos) e o MapBiomas Coleção 11 2025. Ajuste os caminhos em `config/config.yaml`.
```bash
cd scripts
python 01_pam_censo.py           # PAM + Censo das 3 culturas (SIDRA/IBGE)
python 02_mapa_unico.py          # mapa único na GPU, em blocos (~7 min); teste: python 02_mapa_unico.py teste 2611101
python 03_documento.py           # tabela final + docs/CLASSES_MAPA_UNICO.md
```
Requisitos: Python 3.11, torch (CUDA opcional; roda na CPU), rasterio, geopandas, pandas, pyyaml, requests.
