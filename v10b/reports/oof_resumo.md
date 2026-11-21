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

## unet_conc_b

- Limiar global (máx. IoU agregado): **0.4**
- Agregado fora do fold: IoU **0.804** | F1 0.891 | P 0.904 | R 0.879
- Por fold no limiar global: IoU médio 0.752 ± 0.142
- Épocas fixas p/ modelo final (mediana da melhor época): **8**

|   fold |   iou |    f1 |     p |     r |
|-------:|------:|------:|------:|------:|
|      1 | 0.869 | 0.93  | 0.981 | 0.884 |
|      2 | 0.897 | 0.945 | 0.935 | 0.956 |
|      3 | 0.805 | 0.892 | 0.886 | 0.898 |
|      4 | 0.677 | 0.807 | 0.841 | 0.776 |
|      5 | 0.511 | 0.677 | 0.543 | 0.898 |
