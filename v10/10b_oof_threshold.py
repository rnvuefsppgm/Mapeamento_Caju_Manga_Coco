"""10b (v9) — Limiar GLOBAL pelas predições fora do fold + métricas agregadas + épocas fixas.

Uso: python 10b_oof_threshold.py unet_emb_B unet_emb_A  -> reports/oof_resumo.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
TH = np.round(np.arange(0.05, 0.96, 0.05), 2)


def metrics(p, t, th):
    m = t != 255
    pr, tt = p[m] >= th, t[m] == 1
    tp, fp, fn = (pr & tt).sum(), (pr & ~tt).sum(), (~pr & tt).sum()
    P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"iou": tp / max(tp + fp + fn, 1), "f1": 2 * P * R / max(P + R, 1e-9), "p": P, "r": R}


out = ["# Limiar global (predições fora do fold) — v9", ""]
for name in sys.argv[1:]:
    folds = sorted((ROOT / "reports" / "oof").glob(f"{name}_fold*.npz"))
    if not folds:
        continue
    data = [np.load(f) for f in folds]
    P = np.concatenate([d["p"].astype("float32").ravel() for d in data])
    T = np.concatenate([d["t"].ravel() for d in data])
    curve = {th: metrics(P, T, th) for th in TH}
    best = max(curve, key=lambda k: curve[k]["iou"])
    jf = ROOT / "reports" / f"spatial_cv_{name}.json"  # C1/C4 não têm (épocas n/d)
    cv = json.loads(jf.read_text(encoding="utf-8")) if jf.exists() else {"per_fold": []}
    ep = [r.get("cnn_best_epoch") for r in cv["per_fold"] if r.get("cnn_best_epoch") is not None]
    rows = []
    for f, d in zip(folds, data):
        m = metrics(d["p"].astype("float32").ravel(), d["t"].ravel(), best)
        rows.append({"fold": f.stem.split("_fold")[-1], **{k: round(v, 3) for k, v in m.items()}})
    g = curve[best]
    (ROOT / "reports" / f"oof_{name}.json").write_text(json.dumps({"limiar": float(best), "iou": float(g["iou"]),
        "f1": float(g["f1"]), "p": float(g["p"]), "r": float(g["r"]),
        "epocas": int(np.median(ep)) + 1 if ep else None}), encoding="utf-8")  # v10
    out += [f"## {name}", "",
            f"- Limiar global (máx. IoU agregado): **{best}**",
            f"- Agregado fora do fold: IoU **{g['iou']:.3f}** | F1 {g['f1']:.3f} | P {g['p']:.3f} | R {g['r']:.3f}",
            f"- Por fold no limiar global: IoU médio {np.mean([r['iou'] for r in rows]):.3f} ± {np.std([r['iou'] for r in rows]):.3f}",
            f"- Épocas fixas p/ modelo final (mediana da melhor época): **{int(np.median(ep)) + 1 if ep else 'n/d'}**",
            "", pd.DataFrame(rows).to_markdown(index=False), ""]
(ROOT / "reports" / "oof_resumo.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
