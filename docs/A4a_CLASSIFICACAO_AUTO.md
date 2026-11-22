# A4a — Classificação automática dos candidatos (MANGA_ok / NAO_MANGA / DUVIDA)

Script: `scripts_amostras/A4a_classificacao_auto.py` · Parâmetros: `config/config.yaml` → `amostragem.auto_classificacao`
Resultado: `reports/A4a_classificacao_auto.md` · Pedido do usuário em 21/11/2026: separar os candidatos por probabilidade em vez da conferência manual.

## Papel destes rótulos
São **rótulos automáticos** (`origem = "auto"`). Servem para montar o modelo da rodada 2 de candidatos e entram no treino, mas **não servem para avaliar** o mapa (CV, limiar, trava). Se servissem, o modelo seria julgado pelos próprios critérios e a avaliação sairia otimista. A avaliação continua dependendo de rótulos conferidos visualmente.

## Método
1. **Classificador**: XGBoost na GPU, 4 classes, sobre os 64 embeddings AlphaEarth 2025 de cada pixel.

   | Classe | Origem dos pixels |
   |---|---|
   | 1 manga | 258 polígonos de Livramento (código 1 do raster de campo, borda erodida) |
   | 0 outros | Livramento código 3 (área amostrada, a mais de 100 m dos pomares) + classes seguras MapBiomas {3, 4, 5, 23, 24, 25, 30, 31, 33} nos 15 municípios |
   | 2 caju | polígonos de campo do caju (Serra do Mel, `D:\Modelo_Caju\v7\data\vectors\Pol_Caju.shp`) |
   | 3 coco | `COCO.kmz` do projeto coco |

   Pesos inversos à frequência de cada classe.
2. **Calibração**: CV espacial em 5 folds. Em Livramento, os grupos são blocos de 3 km, para que pomares vizinhos não fiquem um no treino e outro no teste. A p_manga de cada pixel é calculada com o modelo que não o viu e calibrada por regressão isotônica, para que "0,8" signifique cerca de 80% de manga nos dados de validação.
3. **Candidato**: média da p calibrada nos pixels do polígono, mais p_caju e p_coco.
4. **Regras de pasta** (versão 2, **opção A, por pixel**, escolhida pelo usuário em 21/11/2026)

   **Pixel confiável**: p ≥ 0,80 na Chapada (Livramento, Dom Basílio, Ituaçu, Tanhaçu, Rio de Contas, Barra da Estiva), **ou** p ≥ 0,90 e p_manga − max(p_caju, p_coco) ≥ 0,5 nas regiões novas (Vale do S. Francisco, Norte de MG, RN). Depois, abertura de 1 px e descarte de manchas com menos de 25 px.

   | Pasta | Regra |
   |---|---|
   | MANGA_ok | classe MapBiomas compatível {15, 46, 48, 41, 21}, parte confiável ≥ 1 ha **e** ≥ 50% do candidato. **O polígono gravado é só o recorte confiável**, não a mancha inteira |
   | NAO_MANGA | p ≤ 0,20, ou controle negativo com p < 0,50 |
   | DUVIDA | o resto; também o excedente de MANGA_ok acima do teto da PAM |

5. **Teto PAM**: a área MANGA_ok por município fica limitada a 0,6 × a área colhida na PAM 2025 (regra do caju v10a). O excedente, a começar pelos de menor p, vai para DUVIDA.
6. **Checagem por amostragem**: sorteio de até 10 candidatos por pasta, em `rodada1_checagem.kml` (pasta única `CONFERIR`, em ordem aleatória). O usuário move cada um para a pasta certa, e o gabarito (`rodada1_checagem_gabarito.csv`) dá a taxa de acerto de cada pasta. **Critério**: se mais de 10% dos MANGA_ok estiverem errados, apertar os limiares e refazer.

## Histórico
| Versão | Regra MANGA_ok | Resultado |
|---|---|---|
| 1 | p **médio** do candidato ≥ 0,80 (Chapada) / ≥ 0,90 (regiões novas) | 0 MANGA_ok, 131 NAO_MANGA, 92 DUVIDA. Candidatos grandes e mistos têm média baixa (os "alta" de Livramento ficaram com 0,63), e a calibração quase não passa de 0,93 |
| 2 (opção A) | p **por pixel** + recorte confiável (acima) | 3 MANGA_ok (1.128 ha, todos em Livramento), 131 NAO_MANGA, 89 DUVIDA. Os 3 MANGA_ok são 88–95% os próprios polígonos de treino: nenhuma manga nova. Outros 6 candidatos têm ≥ 1 ha confiável (613 ha) mas < 50% do candidato |

## Limitações
- Toda a manga do treino vem de uma só paisagem (Chapada). No Vale do São Francisco, a probabilidade é extrapolação, por isso a regra lá é mais exigente.
- Candidatos grandes (até 2.000 ha) juntam pomar, estrada e outras culturas. A média de p dilui isso, mas o polígono não é um talhão.
- O coco tem só 2 polígonos de referência, então a separação manga × coco no Vale é fraca.
