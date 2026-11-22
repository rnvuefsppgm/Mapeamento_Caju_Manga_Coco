# Prompt de coleta de amostras — adaptado à manga (base: etapa de amostras do projeto coco)

Original do usuário: repetir a etapa de amostras do coco (`D:\Backup_2026\Geral_desktop\Mapa_COCO`).
Abaixo, a versão ajustada ao que já existe no projeto manga. Mudanças em **negrito**.

## Economia de tokens
Regras em `CLAUDE.md` (inclui as do coco: STATUS primeiro, grep/sed para portar, ≤ 20 linhas, `tools/resumo.py`,
background sem polling). Do coco foram lidos só `STATUS.md`, `CLAUDE.md` e `docs/ESTRATEGIA_ROTULOS.md`.
Scripts desta etapa ficam em `scripts_amostras/` (prefixo A) para não misturar com o fluxo de dados/treino (01–09).

## Etapas
| # | Etapa | Script | O que muda em relação ao prompt original |
|---|---|---|---|
| A1 | Ranking PAM × Censo Agro 2017 × bioma; **mostrar os 15 maiores e parar** | `A1_censo_agro_manga.py` (de `01_censo_agro_coco.py`) | **PAM já feita** (`02_pam_municipios.py`, Manga = 2737). Falta só o Censo (SIDRA 6955, c227, código da manga lido dos metadados da 6955) e a tabela cruzada |
| A2 | Embeddings 2025 | — | **Nada a exportar**: os 25 maiores da Caatinga (que contêm os 15) já estão em `dados/embeddings/` (EPSG:32724, 38 GB). UTM local + `run_export.sh` do coco só se entrar município novo |
| A3 | Amostras-modelo | — | Pasta do Google Earth Pro `MANGA`, KMZ em **`dados/amostras_usuario/`**. Modelo extra: **`dados/amostras/manga_poligonos.gpkg` (258 polígonos únicos)** — o gpkg do caju tem 448 porque repete MANGA_2/3/P1 dentro do MANGA_4 |
| A4 | Candidatos por similaridade | `A3_candidatos_similaridade.py` (de `05_candidatos_similaridade.py`) | Parâmetros mantidos (kNN cosseno k=5; T = p10 da similaridade cruzada entre polígonos; "explorar" T−0,03..T; excluir {3,4,5,23,24,25,30,31,33}; manchas ≥ 1 ha; tercis alta/média/limiar; controle = classe 48 com sim < T−0,10; cotas 30/25/15/20/10%; saídas por lista; KML aberto com `Start-Process`). **Novos:** (a) modelo = até 20 px por polígono (~5 mil px; o coco tinha 597) para caber na GPU; (b) **filtro contrastivo**: candidato só se sim_manga − sim_negativos ≥ 0,02, com protótipos de **caju** (polígonos de Serra do Mel) e **coco** (`COCO.kmz` do projeto coco) — o mapa de caju não serve aqui: não há sobreposição entre os 25 municípios do caju e os da manga, e a v10b ainda não gerou mapas |
| A5 | Conferência no Google Earth Pro | `A4_importar_conferencia.py` | Pastas `MANGA_ok` / `NAO_MANGA` / `DUVIDA` → `dados/rotulos/r_ouro.gpkg`. Rodada seguinte: `MANGA_ok` entra no modelo, `NAO_MANGA` vira protótipo negativo; **modelo por região** (abaixo) |

## Regiões (modelo por região a partir da rodada 2)
| Região | Municípios (dos 25) | Paisagem |
|---|---|---|
| Submédio São Francisco irrigado | Juazeiro, Petrolina, Casa Nova, Belém do S. Francisco, Sta. Maria da Boa Vista, Lagoa Grande, Curaçá, Sento Sé, Abaré, Orocó, Sobradinho, Cabrobó | pomares irrigados de alta densidade; vizinhos uva, coco, banana, goiaba, acerola |
| Chapada Diamantina / Brumado | **Livramento** (amostras), Dom Basílio, Rio de Contas, Ituaçu, Tanhaçu, Barra da Estiva | irrigação do rio Brumado; manga em "pastagem" na MapBiomas (84%) |
| Norte de MG / Médio S. Francisco | Jaíba, Matias Cardoso, Janaúba, Carinhanha | perímetros irrigados (Jaíba, Gorutuba); banana e limão vizinhos |
| Vale do Açu (RN) / Cariri (CE) | Ipanguaçu, Assú, Mauriti | fruticultura irrigada; caju e coco por perto |

## Lista da rodada 1
`top15_caatinga` = 15 maiores produtores da Caatinga (PAM 2023–25), definida após A1. Expectativa: muitos candidatos na
Chapada (paisagem das amostras) e poucos no São Francisco — como no coco (Rodelas/Petrolândia = 0). **Recomendado:
digitalizar 5–10 pomares em Petrolina/Juazeiro/Casa Nova e 3–5 em Jaíba antes da rodada 1.**

## Guia rápido de fotointerpretação (descrição no KML)
- **Manga**: copas arredondadas, densas, verde-escuras, em linhas regulares (~4–10 m), poda em "guarda-chuva" nos pomares do Vale.
- **Não-manga parecido**: uva (latada contínua e plana, sem copa individual); coco (copa em estrela); banana (verde-claro, textura
  grossa, sem copas redondas); caju (copa irregular, sequeiro, espaçamento maior); goiaba/acerola/limão (copas menores).
- Anotar o ano da imagem do Google Earth (`ano_img`).
