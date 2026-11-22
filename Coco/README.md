# Mapeamento_coco

Mapeamento de coco-da-baía 2025 (10 m) nos municípios produtores da Caatinga, com embeddings AlphaEarth (Google
Satellite Embedding) e U-Net ajustada a partir do modelo de caju. Sem dados de campo: os rótulos vêm de amostras
digitalizadas no Google Earth Pro, similaridade espectral, interpretação visual da copa e regras com prior de área
(PAM/IBGE e Censo Agropecuário 2017). Comparação com PAM, Censo, MapBiomas e o mapa global de Descals.

**Como usar:** [TUTORIAL.md](TUTORIAL.md) · **Resultado v1:** [docs/RELATORIO_FINAL_V1.md](docs/RELATORIO_FINAL_V1.md)

## Resultado v1 — 15 maiores produtores da Caatinga NE

| Item | Valor |
|---|---|
| Validação cruzada espacial (5 folds, blocos de 10 km) | IoU 0,789 · F1 0,882 · precisão 0,891 · recall 0,873 |
| Área mapeada | 23.247 ha (PAM 2023–25: 54.136 ha; Censo 2017: 33.325 ha) |
| Concorda com a PAM (razão 0,5–1,3) | Touros, Paraipaba, Petrolândia, Rodelas, Petrolina, Juazeiro |
| Subestimado (litoral CE, poucas amostras de sequeiro) | Trairi, Acaraú, Itarema, Itapipoca, Beberibe, Amontada, Cascavel, Camocim |

## Arquivos de resultado (`resultados/v1/`)

| Arquivo | Conteúdo |
|---|---|
| `coco_2025_top15_caatinga_classe.tif` | mapa único, EPSG:32724, 10 m: 1 = coco, 0 = não coco, 255 = fora dos municípios |
| `coco_2025_top15_caatinga_prob.tif` | probabilidade de coco (0–100) |
| `municipios/<código>_<nome>_{classe,prob}.tif` | mapas por município (UTM local) |
| `vetores/coco_2025_top15_caatinga.{kmz,gpkg}` | polígonos (área e probabilidade média) para Google Earth / SIG; também por município |
| `relatorios/` | comparação por município (md/csv) e resumo da validação cruzada |

## Organização

| Pasta | Conteúdo |
|---|---|
| `config/config.yaml` | todos os caminhos e parâmetros (nenhum caminho fixo no código) |
| `scripts/` | pipeline em ordem numérica (00 → 50) |
| `docs/` | plano, estratégia de rótulos, referências, desempenho, relatórios |
| `tools/` | utilidades (resumo de raster/vetor/tabela) |
