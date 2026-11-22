# Limiar global (predições fora do fold) — v9

## unet_emb_v7

- Limiar global (máx. IoU agregado): **0.5**
- Agregado fora do fold: IoU **0.649** | F1 0.787 | P 0.776 | R 0.799
- Por fold no limiar global: IoU médio 0.594 ± 0.172
- Épocas fixas p/ modelo final (mediana da melhor época): **n/d**

|   fold |   iou |    f1 |     p |     r |
|-------:|------:|------:|------:|------:|
|      1 | 0.74  | 0.85  | 0.9   | 0.806 |
|      2 | 0.772 | 0.871 | 0.853 | 0.89  |
|      3 | 0.661 | 0.796 | 0.714 | 0.899 |
|      4 | 0.481 | 0.65  | 0.688 | 0.616 |
|      5 | 0.315 | 0.479 | 0.346 | 0.782 |

## unet_conc

- Limiar global (máx. IoU agregado): **0.55**
- Agregado fora do fold: IoU **0.661** | F1 0.796 | P 0.785 | R 0.808
- Por fold no limiar global: IoU médio 0.601 ± 0.177
- Épocas fixas p/ modelo final (mediana da melhor época): **28**

|   fold |   iou |    f1 |     p |     r |
|-------:|------:|------:|------:|------:|
|      1 | 0.791 | 0.884 | 0.927 | 0.844 |
|      2 | 0.759 | 0.863 | 0.854 | 0.872 |
|      3 | 0.64  | 0.78  | 0.736 | 0.83  |
|      4 | 0.504 | 0.67  | 0.68  | 0.66  |
|      5 | 0.31  | 0.473 | 0.331 | 0.827 |
