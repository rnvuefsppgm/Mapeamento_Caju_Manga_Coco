# CV espacial (5 folds) — coco v1

Limiar global OOF = **0.65** | IoU 0.789 | F1 0.882 | precisão 0.891 | recall 0.873 (seleção: ouro sobre automático) | só ouro: IoU 0.088, precisão 0.094, recall 0.545 | trava IoU ≥ 0.65: **APROVADO**

| Fold | Melhor época | IoU val@0,5 | IoU@limiar | Precisão | Recall |
|---|---|---|---|---|---|
| oof_fold0 | 59 | 0.577 | 0.593 | 0.712 | 0.78 |
| oof_fold1 | 54 | 0.721 | 0.701 | 0.91 | 0.753 |
| oof_fold2 | 47 | 0.85 | 0.849 | 0.95 | 0.888 |
| oof_fold3 | 49 | 0.873 | 0.876 | 0.916 | 0.953 |
| oof_fold4 | 34 | 0.872 | 0.877 | 0.928 | 0.941 |