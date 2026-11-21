"""10 (v7) — Compara os modelos do CV espacial (mesmos folds) e gera reports/comparacao_v7.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
NAMES = sys.argv[1:] or ["unet_emb", "unet_sentinel", "unet_sentinel_emb"]
KEYS = ["iou_cashew", "f1_cashew", "precision_cashew", "recall_cashew", "threshold", "minutes"]

rows, folds = [], {}
for n in NAMES:
    f = ROOT / "reports" / f"spatial_cv_{n}.json"
    if not f.exists():
        print("ausente:", f.name)
        continue
    pf = pd.DataFrame(json.loads(f.read_text(encoding="utf-8"))["per_fold"])
    folds[n] = pf.set_index("fold")
    r = {"modelo": n, "folds": int(pf[f"cnn_{KEYS[0]}"].notna().sum())}
    for k in KEYS:
        c = f"cnn_{k}"
        if c in pf:
            v = pf[c].dropna()
            r[k] = f"{v.mean():.3f} ± {v.std(ddof=0):.3f}" if k != "minutes" else f"{v.sum():.0f}"
    rows.append(r)

out = ["# Comparação v7 — CV espacial (5 folds, mesmos blocos)", "",
       pd.DataFrame(rows).to_markdown(index=False), "",
       "## IoU do caju por fold", ""]
if folds:
    iou = pd.DataFrame({n: d["cnn_iou_cashew"] for n, d in folds.items()})
    iou["blocos_teste"] = next(iter(folds.values()))["test_blocks"]
    out += [iou.round(3).to_markdown(), ""]
    best = iou.drop(columns="blocos_teste").mean().idxmax()
    wins = (iou.drop(columns="blocos_teste").idxmax(axis=1) == best).sum()
    out += [f"Melhor média de IoU: **{best}** (vence em {wins} de {len(iou)} folds).", ""]
    (ROOT / "reports" / "best_model.txt").write_text(best, encoding="utf-8")
(ROOT / "reports" / "comparacao_v7.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
