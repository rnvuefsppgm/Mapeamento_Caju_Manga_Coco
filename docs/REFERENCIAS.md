# Referências do projeto Mapa de manga

Uma entrada por obra/dado, com **onde é usada**. Acrescente aqui (não repita buscas: ver regra 8 do `CLAUDE.md`).
Detalhes de acesso e caminhos locais dos dados: `docs/FONTES.md`.

## Artigos e documentos
| # | Referência | Uso no projeto |
|---|---|---|
| R4 | BROWN, C. F. et al. AlphaEarth Foundations: an embedding field model for accurate and efficient global mapping from sparse label data. arXiv:2507.22291, 2025. Produto: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` | entrada do modelo (64 embeddings, 10 m, 2025) |
| R5 | OLOFSSON, P. et al. Good practices for estimating area and assessing accuracy of land change. *Remote Sensing of Environment*, 148, 42–57, 2014 | validação independente e estimativa de área com IC (E7) |
| R6 | Projeto Caju (interno): `D:\Modelo_Caju\DECISOES.md`, `VERSOES_SINTESE.md`, `v10\V10_AMOSTRAS_CONCORDANCIA.md`, `v10b\V10B_MAPBIOMAS_100EP.md` | receita de rótulos por concordância, treino e predição |

## Bases de dados
| # | Referência | Uso |
|---|---|---|
| D1 | IBGE. Produção Agrícola Municipal (PAM), tabela SIDRA 1613, Manga (2737), 2023–2025 | ranking, prior/teto de área |
| D2 | IBGE. Censo Agropecuário 2017, tabela SIDRA 6955, manga (código da classificação 227 a confirmar). ibge.gov.br/estatisticas/economicas/agricultura-e-pecuaria/21814-2017-censo-agropecuario.html | área existente, nº de pés, estabelecimentos |
| D3 | IBGE. Malha municipal 2024 (`BR_Municipios_2024`) | limites, regiões de exportação |
| D4 | PROJETO MAPBIOMAS. Coleção 11 da série anual de mapas de cobertura e uso da terra do Brasil, 2025 (30 m) | classes compatíveis e negativos seguros |
| D5 | Google Earth Pro (imagens de alta resolução, várias fontes/datas) | fotointerpretação das amostras (anotar a data da imagem) |

## Ferramentas
| # | Referência | Uso |
|---|---|---|
| T1 | DatensEE 0.1.0a8 (exportação local via EE High Volume API) | exportação dos embeddings |
| T2 | segmentation_models_pytorch, PyTorch 2.14 (CUDA), XGBoost 3.2 (GPU) | modelos |
| R7 | Projeto Coco (interno): `D:\Backup_2026\Geral_desktop\Mapa_COCO\docs\ESTRATEGIA_ROTULOS.md`, `scripts/05_candidatos_similaridade.py` | método de candidatos por similaridade |
