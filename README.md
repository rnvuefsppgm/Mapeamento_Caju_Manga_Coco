# Mapeamento de manga 2025 — Caatinga (embeddings AlphaEarth + U-Net)

Mapa de áreas de **manga (Mangifera indica)** em 2025, a 10 m, nos **25 municípios que mais produzem manga na Caatinga**: 92,5% da produção do bioma e cerca de 76% da produção nacional (PAM/IBGE, média 2023–2025). Repete a estratégia do mapeamento de caju (`Modelo_Caju`, versão v10b) e a etapa de amostras do mapeamento de coco (`Mapa_COCO`).

> **Versão atual: v1 (piloto).** A rede foi treinada só com amostras da Chapada Diamantina/Brumado-BA. Na Chapada o mapa é avaliado; nas demais regiões os valores são **provisórios** (ver [Limitações](#limitações)).

## Resultado v1
| Medida | Valor |
|---|---|
| Validação cruzada espacial (5 folds, blocos de 3 km, só rótulos conferidos) | **IoU 0,694** · F1 0,82 · precisão 0,82 · recall 0,82 (limiar 0,55) |
| Área mapeada (25 municípios) | **26.551 ha** (2.253 manchas) × PAM 2025: 60.129 ha (razão 0,44) |
| Correlação mapa × PAM por município | 0,63 |

| Região | Situação |
|---|---|
| Chapada (Livramento de Nossa Senhora e vizinhos) | avaliada; Livramento = 0,52× PAM (0,64× Censo 2017) |
| Vale do São Francisco (Petrolina, Juazeiro, Casa Nova…) | extrapolação; Casa Nova ≈ Censo, Sento Sé 0,90× PAM, Petrolina 0,54× Censo, Juazeiro 0,18× PAM |
| Norte de MG (Jaíba, Janaúba, Matias Cardoso) | **não confiável**: 2,2× a 8,4× a PAM (confusão com banana e limão irrigados) |
| Vale do Açu-RN e Cariri-CE | quase nada detectado |

Tabela completa: [`reports/v1_comparacao.md`](reports/v1_comparacao.md).

## Dados
| Fonte | Uso |
|---|---|
| **Google Satellite Embedding (AlphaEarth Foundations) 2025**, 64 bandas, 10 m | entrada da rede; exportado com DatensEE (High Volume API), 38 GB |
| **Amostras de campo**: 258 polígonos de manga (1.683 ha) em Livramento de Nossa Senhora-BA (KML/KMZ do usuário) | rótulos conferidos (treino e **avaliação**) |
| **PAM/IBGE** (SIDRA 1613, Manga = 2737), 2023–2025 | seleção dos municípios e comparação de área |
| **Censo Agropecuário 2017** (SIDRA 6955, Manga = 4968) | comparação e alerta PAM × Censo |
| **MapBiomas Coleção 11, 2025** (30 m) | classes seguras de não-manga e máscara de classes compatíveis {15, 46, 48, 41, 21} |
| Malha municipal IBGE 2024 · bioma predominante por município | recortes e seleção |
| Pesos finais da U-Net do caju v10b; polígonos de caju (Serra do Mel-RN) e de coco | transferência de aprendizado e exemplos de não-manga |

## Método
1. **Dados**: amostras KML → GeoPackage sem duplicatas; ranking PAM × Censo × bioma; regiões (município + 1 km); embeddings 2025; recorte MapBiomas a 10 m; raster de rótulos de campo em códigos (manga, faixa ignorada, negativo da área amostrada, negativo MapBiomas, desconhecido).
2. **Amostras (método do coco)**:
   - candidatos por similaridade às amostras (kNN de cosseno, k = 5, limiar = percentil 10 da similaridade cruzada entre polígonos, filtro contra caju e coco);
   - classificação automática com XGBoost em 4 classes (manga, outros, caju, coco), calibrada por validação cruzada espacial, AUC 0,957;
   - mapa automático por pixel com a mesma regra (pixel confiável, manchas de 1 ha ou mais).
3. **Rótulos da rede (receita caju v10b)**:
   - manga = polígonos conferidos + manchas automáticas de alta confiança;
   - não-manga = classes seguras MapBiomas + probabilidade baixa + candidatos NAO_MANGA + área amostrada;
   - anel de 2 px ignorado em volta da manga.
4. **Rede**: U-Net (ResNet-18), 64 canais.
   - Três proteções na transferência do caju:
     1. cabeça reiniciada;
     2. 60 chips de caju como não-manga;
     3. teste A/B na validação cruzada: pesos do caju 0,694 × ImageNet 0,302.
   - Validação cruzada espacial avaliada **só nos rótulos conferidos**; trava IoU ≥ 0,60.
   - Modelo final: 5 sementes × 100 épocas, aquecimento, encoder lento, EMA e perda focal + dice + Lovász.
5. **Predição**: 5 membros, TTA ×2, limiar 0,55, máscara MapBiomas de classes compatíveis, manchas e buracos menores que 1 ha removidos, recorte municipal.

## Estrutura
```
config/config.yaml          todos os parâmetros (nada de caminho fixo no código)
00_status.py                estado do projeto, uma linha por etapa
01–09_*.py, 04_*.sh         dados: amostras, PAM, regiões, embeddings, VRT, MapBiomas, rótulos de campo, chips, cache
scripts_amostras/A*.py      amostras: Censo, candidatos, classificação automática, mapa automático
10–19_*.py                  rede v1: rótulos, chips, pesos iniciais, CV, limiar, final, predição, mosaico, comparação, vetores
run_dados.sh, run_treino_v1.sh, run_pos_treino_v1.sh   cadeias retomáveis
src/                        biblioteca do caju v10b (dataset, modelos, treino, predição)
docs/                       plano, estratégia de amostras, métodos, referências
reports/                    relatórios (md/csv)
STATUS.md · DECISOES.md · CLAUDE.md    estado atual, registro de decisões, regras de trabalho
```
Dados brutos, modelos e logs (`dados/`, `modelos/`, `logs/`) ficam fora do git (ver `.gitignore`); os mapas finais estão em `resultados/`.

## Produtos
**Neste repositório**, em [`resultados/`](resultados/):

| Produto | Arquivo |
|---|---|
| Mapa único dos 25 municípios (classe: 1 = manga, 0 = não, 255 = fora) | `resultados/mapa_25municipios/manga_2025_25mun_classe.tif` |
| Mapas por município (classe) | `resultados/municipios/<código>_<nome>_class.tif` (25) |
| Vetores para Google Earth / GIS (área em ha e probabilidade média por mancha) | `resultados/vetores/manga_2025_25mun.{kmz,gpkg}` e `<código>_<nome>_manga.{kmz,gpkg}` |
| Tabela mapa × PAM × Censo | `resultados/v1_comparacao.md` |

Os rasters de **probabilidade** (0–100, cerca de 100 MB) e os pesos dos modelos ficam fora do git por causa do tamanho. Eles são gerados pelas etapas 16 e 15 em `dados/predicoes/` e `modelos/`.

Todos em EPSG:32724 (UTM 24S), com pixel de 10 m.

**Tutorial de uso dos scripts: [`TUTORIAL.md`](TUTORIAL.md).**

## Como reproduzir
Ambientes: `D:/Modelo_Caju/.venv` (Python 3.11, PyTorch CUDA, XGBoost GPU, geopandas, rasterio) e `D:/Modelo_Caju/.venv_v8` (DatensEE).
```bash
python 00_status.py                       # estado
bash run_dados.sh                         # 04 exportação -> 05..09
python scripts_amostras/A1_censo_agro_manga.py
python scripts_amostras/A3_candidatos_similaridade.py top15_caatinga
python scripts_amostras/A4a_classificacao_auto.py top15_caatinga
python scripts_amostras/A5_mapa_xgb.py top15_caatinga
bash run_treino_v1.sh                     # 10..14: rótulos, chips, CV A/B, trava, final
bash run_pos_treino_v1.sh                 # 16..19: predição, mosaico, comparação, vetores
```
GPU usada: RTX 3050 com 4 GB. Com outro treino na mesma GPU, use `GPU_FRAC` (fração da memória).

## Limitações
- **Avaliação só em Livramento**: o IoU de 0,694 vale para a Chapada; o Vale do São Francisco, o Norte de MG e o RN/CE não têm rótulo conferido.
- **Norte de MG superestimado** (banana e limão irrigados) e **Vale subestimado** em vários municípios.
- Negativos da área amostrada supõem que tudo fora dos polígonos, perto deles, não é manga.
- O mapa mede área plantada; a PAM mede área colhida declarada. A PAM de Belém do São Francisco, Lagoa Grande e Tanhaçu diverge muito do Censo 2017.

## Próximos passos (v2)
Amostras conferidas no Vale do São Francisco e no Norte de MG; modelo ou rótulos por região; controle de qualidade com cerca de 200 pontos conferidos (precisão de manga ≥ 0,90).

## Referências
Ver [`docs/REFERENCIAS.md`](docs/REFERENCIAS.md), entre elas Brown et al. (AlphaEarth Foundations), Olofsson et al. (2014), PAM/IBGE, Censo Agro 2017 e MapBiomas Coleção 11. Histórico de decisões em [`DECISOES.md`](DECISOES.md).
