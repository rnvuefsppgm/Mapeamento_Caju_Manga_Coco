# Recomendações para a próxima versão de treino (guardadas em 20/11/2026)

Contexto: o CV espacial do `unet_emb` (v7, 2025, Serra do Mel) deu **IoU 0,563 ± 0,156** (folds: 0,710 / 0,725 / 0,602 / 0,468 / 0,311). O recall foi alto em todos os folds (0,70–0,93); a precisão caiu nos folds 4 e 5 (0,58 e 0,35). O limiar por fold oscilou de 0,15 a 0,60.

## 1. Treino único guiado pelo CV
- Limiar **global** escolhido nas predições **fora do fold** (todos os pixels de teste dos 5 folds juntos), em vez de um limiar por fold escolhido num único bloco de validação.
- Informar também o IoU desse conjunto todo (mais estável que a média dos folds).
- **Épocas fixas** = mediana da melhor época nos folds (sem parada antecipada num bloco ruidoso).
- **Modelo final** treinado com os 14 blocos, com 3 sementes combinadas em ensemble.

## 2. Mais variação sem coletar no campo
- Embeddings de **2023, 2024 e 2025** com os mesmos polígonos (caju é perene).
- **Faixa de exclusão** de 20–30 m (ignorar = 255) em volta dos polígonos.
- **Negativos confiáveis** da MapBiomas (água, área urbana, dunas, formação florestal) nos municípios vizinhos.
- Teste **com e sem** os pseudo-rótulos da v4.

## 3. Esforço humano curto
- Revisar os 450 pontos de `v7/data/labels/review_points.gpkg` no QGIS.
- Inspecionar os blocos dos folds 4 e 5 (blk_71_941, 71_943, 72_943, 70_941, 70_943) atrás de caju sem polígono.

## 4. Modelo mais simples
- MLP ou LightGBM por pixel com a média e o desvio de uma janela 3×3 dos embeddings, nos mesmos folds. Se empatar com a U-Net, roda direto no Earth Engine para a Caatinga.
