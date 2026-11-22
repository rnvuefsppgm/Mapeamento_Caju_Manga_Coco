"""31 — Limiar GLOBAL pelas predições fora do fold (OOF) + métricas agregadas e por macro-região.

Lê v1/reports/oof_fold*.npz (todo pixel coco + 5 % do não-coco de avaliação; o não-coco é re-ponderado ×20).
Critério: maior IoU agregado. Trava: IoU ≥ treino.trava_iou. Saída: v1/reports/cv_resumo.{json,md}
"""
import json

import numpy as np

from common import cfg, ver

c = cfg()
rep = ver(c, "reports", "x").parent
fs = sorted(rep.glob("oof_fold*.npz"))
P, Y, info = [], [], []
for f in fs:
    d = np.load(f)
    P.append(d["p"].astype(np.float32)); Y.append(d["y"])
    info.append((f.stem, int(d["best_epoch"]), float(d["iou"])))
p, y = np.concatenate(P), np.concatenate(Y)
w = np.where(y == 1, 1.0, 20.0)   # desfaz a subamostragem de 5 % do não-coco


def met(t, p=p, y=y, w=w):
    pr = p >= t
    tp, fp, fn = (w * (pr & (y == 1))).sum(), (w * (pr & (y == 0))).sum(), (w * (~pr & (y == 1))).sum()
    return dict(limiar=round(float(t), 2), iou=tp / max(1, tp + fp + fn), f1=2 * tp / max(1, 2 * tp + fp + fn),
                precisao=tp / max(1, tp + fp), recall=tp / max(1, tp + fn))


ts = np.arange(0.05, 0.96, 0.05)
melhor = max((met(t) for t in ts), key=lambda r: r["iou"])
folds = [dict(fold=n, melhor_epoca=e, iou_val_05=round(i, 3), **{k: round(v, 3) for k, v in
         met(melhor["limiar"], *(lambda d: (d["p"].astype(np.float32), d["y"], np.where(d["y"] == 1, 1.0, 20.0)))(np.load(rep / f"{n}.npz"))).items()})
         for n, e, i in info]
YO = np.concatenate([np.load(f)["y_ouro"] for f in fs]); vo = YO != 255
so = met(melhor["limiar"], p[vo], YO[vo], np.where(YO[vo] == 1, 1.0, np.where(y[vo] == 1, 1.0, 20.0)))
res = {**{k: round(v, 3) if isinstance(v, float) else v for k, v in melhor.items()},
       "trava_iou": c["treino"]["trava_iou"], "aprovado": bool(melhor["iou"] >= c["treino"]["trava_iou"]),
       "so_ouro": {k: round(float(v), 3) for k, v in so.items()}, "epoca_mediana": int(np.median([e for _, e, _ in info])), "folds": folds}
(rep / "cv_resumo.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
linhas = ["# CV espacial (5 folds) — coco v1", "",
          f"Limiar global OOF = **{res['limiar']}** | IoU {res['iou']:.3f} | F1 {res['f1']:.3f} | precisão {res['precisao']:.3f} | "
          f"recall {res['recall']:.3f} (seleção: ouro sobre automático) | só ouro: IoU {res['so_ouro']['iou']:.3f}, precisão {res['so_ouro']['precisao']:.3f}, recall {res['so_ouro']['recall']:.3f} | trava IoU ≥ {res['trava_iou']}: **{'APROVADO' if res['aprovado'] else 'REPROVADO'}**", "",
          "| Fold | Melhor época | IoU val@0,5 | IoU@limiar | Precisão | Recall |", "|---|---|---|---|---|---|"]
linhas += [f"| {f['fold']} | {f['melhor_epoca']} | {f['iou_val_05']} | {f['iou']} | {f['precisao']} | {f['recall']} |" for f in folds]
(rep / "cv_resumo.md").write_text("\n".join(linhas), encoding="utf-8")
print("\n".join(linhas))
