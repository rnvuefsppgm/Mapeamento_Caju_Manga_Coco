# Plano — Treino da rede com as amostras mais seguras (Chapada) e predição

Pedido do usuário (21/11/2026): treinar a rede só com as amostras dos municípios em que as estimativas são melhores e, em seguida, fazer a predição. **Aprovado e em execução (21/11/2026 ~18:15)** com as 3 proteções abaixo.

## Avaliação da proposta
| Ponto | Avaliação |
|---|---|
| A favor | Os rótulos da Chapada são os mais seguros que existem: 258 polígonos conferidos (1.683 ha) + 674 ha automáticos de alta confiança (Livramento 575, Dom Basílio 86). Dá um primeiro mapa rápido, e a rede aproveita o contexto espacial que o XGB por pixel não usa. É o mesmo caminho do caju v7/v8: treinar em Serra do Mel e prever nos 25 |
| Contra | A rede herda o limite das amostras. O caju v8, treinado só em Serra do Mel, errou nas paisagens não vistas (0,61× a PAM no total; 2,2× a 3,6× no PI/PE). Para a manga, espera-se **mapa bom na Chapada e pouco confiável no Vale e no Norte de MG**, onde está cerca de 75% da área |
| Conclusão | Vale como **versão 1 (piloto)**: mapa confiável para a Chapada e sinal provisório nas demais regiões, com aviso no relatório. O mapa do Vale e de MG continua dependendo de amostras de lá. A saída da v1 também serve de fonte para os rótulos automáticos da v2 (como a U-Net v8 serviu ao caju v10) |

## O que será executado
| # | Etapa | Script (porte do caju v10b, com src/ já copiado) | Detalhe |
|---|---|---|---|
| T1 | Rótulos da Chapada | `10_rotulos_chapada.py` (novo, curto) | **Manga** = código 1 de Livramento + manchas confiáveis A5 (Livramento, Dom Basílio). **Não-manga** = Livramento código 3 + classes seguras MapBiomas + p_XGB ≤ 0,05 em classe compatível + NAO_MANGA da A4a. **Ignorado** = resto + anel de 2 px. Municípios: Livramento, Dom Basílio, Ituaçu, Tanhaçu |
| T2 | Chips + cache | `08_chips.py` (stride 64) + `11_chips_chapada.py` + `09_cache.py` | Campo: ~250 chips (era 110 com stride 96). Demais: até 30 com manga + 30 sem por município. Cache 8 bits |
| T3 | CV espacial 5 folds | `12_cv.py` (de `08_spatial_cv.py`) + `13_limiar_oof.py` (de `10b_oof_threshold.py`) | U-Net resnet18, 64 canais, **ajuste fino a partir dos pesos finais do caju v10b** (mesma entrada). Blocos de 3 km; **avaliação só nos 258 polígonos conferidos**; limiar global fora do fold. Até 60 épocas, paciência 15 |
| ⏸ | Trava | – | Segue só se o IoU do CV for ≥ 0,60; senão, paro e reporto |
| T4 | Treino final | `14_final.py` (de `14_final_train.py`) | Todos os chips, 100 épocas × 5 sementes, aquecimento 5, encoder lento 10, EMA, + Lovász (receita v10b) |
| T5 | Predição | `15_predicao.py` + `16_mosaico.py` (de `15_predict_final.py`, `32_mosaico.py`) | **25 municípios** (embeddings prontos), TTA ×2, fora das classes compatíveis MapBiomas = 0, manchas ≥ 1 ha; raster único (prob + classe) |
| T6 | Comparação | `17_comparacao.py` (de `16_comparacao.py`) | Por município: mapa × PAM 2025 × Censo 2017 × A5; rótulo **"confiável"** (Chapada) ou **"extrapolação"** (Vale, MG, RN, CE) |

Tempo estimado (GPU livre): T1–T2 ~15 min · T3 ~30–40 min · T4 ~40–60 min · T5–T6 ~40–60 min → **~2,5–3 h**, numa cadeia só (`run_treino_v1.sh`), retomável.

## Economia de tokens nesta fase
- Porte dos scripts do caju com `sed` de caminhos, nomes e parâmetros; leitura só das linhas que mudam (`grep -n`).
- Uma cadeia em segundo plano com log por etapa; acompanho só pelas notificações. Na trava e no fim, um resumo de ≤ 20 linhas.
- Relatórios em tabela: `reports/T3_cv.md`, `reports/T6_comparacao.md`; `STATUS.md` atualizado ao fim.

## Decisões padrão (mudar se quiser)
- Predição nos **25** municípios (não só nos 15).
- **5 sementes** × 100 épocas (receita v10b).
- Trava **IoU ≥ 0,60** (o caju usou 0,65 em Serra do Mel).

## Proteções do uso dos pesos do caju (aprovadas)
1. **Cabeça reiniciada**: do caju v10b (final_seed42) só encoder e decoder; a última camada começa do zero (`12_init_pesos.py`).
2. **Caju como não-manga**: 60 chips de campo do caju (Serra do Mel, os de maior fração de caju) com todo pixel rotulado = 0.
3. **Teste A/B no CV**: A = caju sem cabeça × B = ImageNet; fica o de maior IoU fora do fold nos rótulos conferidos.

## Scripts efetivos (numeração final)
`10_rotulos_chapada.py` → `08_chips.py` (stride 64) → `11_chips_v1.py` → `12_init_pesos.py` → `13_cv.py` (cópia do caju 08, normalização da manga)
→ `14_limiar_oof.py` → `15_final.py` → `16_predicao.py` → `17_mosaico.py` (GeoTIFF esparso) → `18_comparacao.py`; cadeia em `run_treino_v1.sh`.
Chips: 252 campo (18 blocos de 3 km, avaliação) + 125 automáticos (Chapada) + 60 caju = 437.
