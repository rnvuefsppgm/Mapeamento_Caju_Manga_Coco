# Relatório de parada antes do treino — Mapa de coco v1 (21/11/2026)

Recorte: 15 maiores produtores do NE com Caatinga predominante. Entrada: embeddings AlphaEarth 2025 (64 canais, 10 m).
Dados em `D:\Modelo_Coco\v1`. Tudo pronto para a E5 (treino); **nada foi treinado**.

## 1. Rótulos
| Camada | Origem | Volume |
|---|---|---|
| Ouro (verificado) | 2 polígonos do usuário + 43 candidatos conferidos pela copa + 89 pontos dos QC cegos | 24.568 px coco / 4.567 px não-coco nos chips de campo |
| Automático (treino) | XGBoost regional (litoral / São Francisco) + ranking com prior de área (0,6 × max(PAM, Censo)) + MapBiomas + mapa de caju v10 como negativo | coco 1,1 % · não-coco 69,4 % · ignorado 29,4 % dos pixels dos chips |

**Controle de qualidade (QC cego, 108 pontos, Esri ~1,2 m):** rodada 1 reprovada (coco 0,63) → correções v1b → rodada 2 **aprovada: coco 0,93** (litoral 0,92; SF 0,94), **não-coco 1,00**. 16 pontos de coco ficaram em dúvida (estradas entre quadras, nuvem, quadras escuras manga × coco).

## 2. Coco rotulado × referência (ha, dentro do município)
| Município | Coco rotulado | Ref. max(PAM, Censo) | Razão |
|---|---|---|---|
| Petrolândia | 2.393 | 3.717 | 0,64 |
| Rodelas | 1.445 | 2.370 | 0,61 |
| Petrolina | 1.134 | 2.133 | 0,53 |
| Paraipaba | 2.651 | 6.109 | 0,43 |
| Casa Nova | 230 | 557 | 0,41 |
| Touros | 1.798 | 6.633 | 0,27 |
| Juazeiro | 516 | 2.020 | 0,26 |
| Acaraú | 503 | 6.806 | 0,07 |
| Trairi, Amontada, Itarema, Itapipoca, Beberibe, Cascavel, Camocim | 0–268 | 1.231–8.282 | ≤ 0,03 |
Rótulo de treino é **parcial por desenho** (teto de 0,6 × ref.; o resto fica ignorado, não vira não-coco).

## 3. Chips + cache (`v1/data/cache/`)
| | Litoral | São Francisco |
|---|---|---|
| Chips 256 px | 989 | 619 |
| … com rótulo ouro | 137 | 126 |
| … com ouro coco (blocos de 10 km) | 69 (21 blocos) | 55 (25 blocos) |
Total 1.608 chips, 6,8 GB uint8 (formato do caju v10b), 92 % de pixels válidos em média. Arquivos: `features.npy`, `labels.npy` (treino), `labels_ouro.npy` (avaliação), `chips_metadata.parquet`.

## 4. Limitações
1. **Litoral fora de Paraipaba/Touros/Acaraú quase sem coco rotulado** (o XGBoost do litoral aprendeu a paisagem de Paraipaba). O modelo pode subestimar Trairi, Itarema, Amontada, Itapipoca, Beberibe, Cascavel e Camocim.
2. Avaliação com poucos pixels de **não-coco ouro** (4,6 mil) → precisão do modelo pouco medida só com ouro.
3. Confusão residual coco × manga em quadras de copa escura (São Francisco).
4. Mapa Descals (2020) quase não detecta coco aqui (0 ha no SF; 406 ha em Paraipaba) → só comparação.

## 5. Proposta para o treino (E5) — aguardando sua autorização
- U-Net (receita caju v10b) **iniciada com os pesos do caju**; perda focal+dice (+ Lovász), EMA, aumento de dados na GPU.
- CV espacial 5 folds por blocos de 10 km, estratificado por macro-região; **avaliação** = ouro + não-coco automático de evidência forte (QC 1,00) — recomendado por causa do item 2.
- Limiar global fora do fold; trava IoU ≥ 0,65; modelo final 5 sementes × 100 épocas; ~4–8 h de GPU (compartilhada).
- Alternativa antes do treino (~1 h): **rodada 2 de amostras no litoral** (modelo kNN por município fraco, como foi feito no SF) para reduzir a limitação 1.
