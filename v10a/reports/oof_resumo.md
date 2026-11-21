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

## unet_conc_a

- Limiar global (máx. IoU agregado): **0.55**
- Agregado fora do fold: IoU **0.675** | F1 0.806 | P 0.790 | R 0.823
- Por fold no limiar global: IoU médio 0.611 ± 0.181
- Épocas fixas p/ modelo final (mediana da melhor época): **2**

|   fold |   iou |    f1 |     p |     r |
|-------:|------:|------:|------:|------:|
|      1 | 0.813 | 0.897 | 0.919 | 0.876 |
|      2 | 0.769 | 0.869 | 0.862 | 0.876 |
|      3 | 0.646 | 0.785 | 0.742 | 0.832 |
|      4 | 0.511 | 0.677 | 0.681 | 0.673 |
|      5 | 0.316 | 0.48  | 0.343 | 0.798 |
