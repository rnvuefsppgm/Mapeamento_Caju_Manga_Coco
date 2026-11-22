# Limiar global (predições fora do fold) — v9

## v1_A

- Limiar global (máx. IoU agregado): **0.55**
- Agregado fora do fold: IoU **0.694** | F1 0.819 | P 0.816 | R 0.822
- Por fold no limiar global: IoU médio 0.708 ± 0.129
- Épocas fixas p/ modelo final (mediana da melhor época): **53**

|   fold |   iou |    f1 |     p |     r |
|-------:|------:|------:|------:|------:|
|      1 | 0.619 | 0.765 | 0.666 | 0.898 |
|      2 | 0.501 | 0.668 | 0.51  | 0.968 |
|      3 | 0.755 | 0.861 | 0.994 | 0.759 |
|      4 | 0.829 | 0.907 | 0.96  | 0.859 |
|      5 | 0.834 | 0.91  | 0.913 | 0.907 |

## v1_B

- Limiar global (máx. IoU agregado): **0.45**
- Agregado fora do fold: IoU **0.302** | F1 0.463 | P 0.308 | R 0.936
- Por fold no limiar global: IoU médio 0.219 ± 0.176
- Épocas fixas p/ modelo final (mediana da melhor época): **10**

|   fold |   iou |    f1 |     p |     r |
|-------:|------:|------:|------:|------:|
|      1 | 0.242 | 0.39  | 0.264 | 0.747 |
|      2 | 0.106 | 0.191 | 0.106 | 0.999 |
|      3 | 0.541 | 0.702 | 0.541 | 1     |
|      4 | 0.028 | 0.055 | 0.065 | 0.047 |
|      5 | 0.177 | 0.301 | 0.177 | 0.999 |
