# Referências do projeto Mapa de coco

Uma entrada por obra/dado, com **onde é usada**. Acrescente aqui (não repita buscas: ver regra 8 do `CLAUDE.md`).
Detalhes de acesso e caminhos locais dos dados: `docs/FONTES.md`.

## Artigos e documentos
| # | Referência | Uso no projeto |
|---|---|---|
| R1 | DESCALS, A. et al. High-resolution global map of closed-canopy coconut palm. *Earth System Science Data*, 15, 3991–4010, 2023. doi:10.5194/essd-15-3991-2023. Dados: doi:10.5281/zenodo.8128183 | semente de positivos (F1); comparação final. U-Net em S1 (VV/VH) + S2 (B4), 20 m, 2020; validação com 10.186 pontos |
| R2 | NEVES, R. D.; REIS, J. N. P.; CRUZ, M. P. M.; CRUZ, C. M. Análise da comercialização do coco-verde no estado do Ceará nos anos de 2013 a 2019. *Revista de Economia e Agronegócio*, 20(1), 2022. repositorio.ufc.br/bitstream/riufc/71734/1/2022_art_rdneves.pdf | polos produtores do CE (Fig. 1) → estratos de amostragem; coqueiro em tabuleiros até ~20 km da costa |
| R3 | CAVALCANTE (2015, 2016), citado em R2 | base da Fig. 1 de R2 (configuração espacial do coco no CE) — não consultado diretamente |
| R4 | BROWN, C. F. et al. AlphaEarth Foundations: an embedding field model for accurate and efficient global mapping from sparse label data. arXiv:2507.22291, 2025. Produto: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` | entrada do modelo (64 embeddings, 10 m, 2025) |
| R5 | OLOFSSON, P. et al. Good practices for estimating area and assessing accuracy of land change. *Remote Sensing of Environment*, 148, 42–57, 2014 | validação independente e estimativa de área com IC (E7) |
| R6 | Projeto Caju (interno): `D:\Modelo_Caju\DECISOES.md`, `VERSOES_SINTESE.md`, `v10\V10_AMOSTRAS_CONCORDANCIA.md`, `v10b\V10B_MAPBIOMAS_100EP.md` | receita de rótulos por concordância, treino e predição |

## Bases de dados
| # | Referência | Uso |
|---|---|---|
| D1 | IBGE. Produção Agrícola Municipal (PAM), tabela SIDRA 1613, coco-da-baía (2727), 2023–2025 | ranking, prior/teto de área |
| D2 | IBGE. Censo Agropecuário 2017, tabela SIDRA 6955, coco-da-baía (4945). ibge.gov.br/estatisticas/economicas/agricultura-e-pecuaria/21814-2017-censo-agropecuario.html | área existente, nº de pés, estabelecimentos |
| D3 | IBGE. Malha municipal 2024 (`BR_Municipios_2024`) | limites, regiões de exportação |
| D4 | PROJETO MAPBIOMAS. Coleção 11 da série anual de mapas de cobertura e uso da terra do Brasil, 2025 (30 m) | classes compatíveis e negativos seguros |
| D5 | Google Earth Pro (imagens de alta resolução, várias fontes/datas) | fotointerpretação das amostras (anotar a data da imagem) |
| D6 | Esri World Imagery (ArcGIS Online), tiles z17–z18 (~1,2–0,6 m) — "Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community" | interpretação visual da copa pelo Claude (folhas PNG; cache só para visualização em `dados_externos/esri_cache`) |

## Ferramentas
| # | Referência | Uso |
|---|---|---|
| T1 | DatensEE 0.1.0a8 (exportação local via EE High Volume API) | exportação dos embeddings |
| T2 | segmentation_models_pytorch, PyTorch 2.14 (CUDA), XGBoost 3.2 (GPU) | modelos |
