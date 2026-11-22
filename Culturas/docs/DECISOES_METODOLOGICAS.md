# Decisões metodológicas — mapa único caju + manga + coco 2025

Registro das escolhas feitas na montagem do mapa único, com motivo, alternativa descartada e efeito no resultado.
Resumo em uma linha por decisão: `DECISOES.md` (raiz). Parâmetros: `config/config.yaml`.

| # | Decisão | Motivo | Alternativa descartada | Efeito no resultado | Quem / data |
|---|---|---|---|---|---|
| 1 | Projeto próprio em `D:\Mapeamento_Culturas` (scripts, config, dados, saídas, docs) | produto combina 3 projetos; não misturar com caju/manga/coco | dentro do projeto coco | nenhum nos números | Usuário, 22/11/2026 |
| 2 | Códigos novos **97 caju, 98 manga, 99 coco** | não constam da legenda MapBiomas Coleção 11 (5 abas da planilha conferidas) | 1/2/3 (colidiriam com classes MapBiomas) | mapa pode ser lido junto com produtos MapBiomas | Usuário pediu; valores escolhidos pelo Claude, 22/11/2026 |
| 3 | **Fundo = MapBiomas Coleção 11 2025**, reamostrado de 30 m para 10 m por vizinho mais próximo | mapa único "MapBiomas + culturas" | fundo 0 = outras coberturas | bordas do fundo em degraus de 30 m; culturas mantêm 10 m | Usuário, 22/11/2026 |
| 4 | Sobreposição: vence a maior **margem normalizada** `(p − limiar)/(1 − limiar)` | os modelos têm limiares diferentes (caju 0,30/0,40, manga 0,55, coco 0,65); pela probabilidade bruta o caju venceria quase sempre | probabilidade bruta (pedido inicial) | disputas decididas de forma equilibrada; raster de disputas guardado para conferência | Proposta do Claude, aprovada pelo usuário, 22/11/2026 |
| 5 | **Caju: PAM e Censo de castanha de caju** (SIDRA 1613 produto 2725; 6955 produto 4936) | mesma base usada no projeto caju (PAM castanha) | caju fruto (1613/40473; 6955/4937) | área PAM/Censo do caju comparável ao relatório do caju | Claude, 22/11/2026 |
| 6 | **Limiar do caju por município**: 0,30 no sertão CE/RN (Alto Santo, Aracoiaba, Barreira, Chorozinho, Ocara, Pacajus, Apodi, Cerro Corá, Serra do Mel, Severiano Melo); 0,40 nos demais | regra do mapa final caju v10b: no sertão a probabilidade é a fusão U-Net + XGBoost (limiar 0,30); no resto, só U-Net (0,40) | limiar único 0,40 | margem do caju correta em cada região | Claude, 22/11/2026 |
| 7 | Cultura só conta nos municípios em que foi mapeada (caju 25, manga 25, coco 15) | evitar ler "não mapeado" como "ausente" | tratar todo o recorte como mapeado | tabela mostra "—" onde a cultura não foi mapeada; `cobertura_municipios.csv` | Claude, 22/11/2026 |
| 8 | PAM = área destinada à colheita, **média 2023–25**; Censo = **área existente 2017** | mesmas referências dos projetos individuais | só 2025 / área colhida | comparação consistente com os relatórios de cada cultura | Claude, 22/11/2026 |
| 9 | Pixels com probabilidade > 100 tratados como 100 | aparecem só nas pirâmides (média com NoData); resolução cheia vai até 100 | — | nenhum na prática | Claude, 22/11/2026 |
| 10 | Processamento na **GPU** (torch), em blocos de 4096 px, pulando blocos fora dos municípios | pedido do usuário; decisão e contagem de áreas numa só passada | CPU | ~1,3 s por bloco | Usuário, 22/11/2026 |

## Limitações herdadas dos mapas de origem
- **Coco**: litoral do CE fora de Paraipaba subestimado (razão mapa/PAM ≤ 0,10) — poucas amostras de coqueiral de sequeiro.
- **Caju e manga**: ver os relatórios de cada projeto (`D:\Modelo_Caju\v10b\reports`, `MAPA_MANGA\reports`).
- Uma cultura subestimada perde disputas só onde foi marcada; não ganha área com o mapa único.
