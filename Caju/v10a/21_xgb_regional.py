"""21 (v10a) — XGBoost REGIONAL na GPU (3 regiões), treinado nos municípios "bons" da v10.

Bons = razão v10/PAM entre 0,5 e 2,0. Positivos = pixels v10 >= 0,80 (+ polígonos de campo de Serra do Mel,
erodidos); negativos = pixels v10 <= 0,05. Cada região usa seus bons municípios + 30% de amostra dos bons das
outras regiões (garante dados no PI/PE). Atributos = 64 embeddings + média 3x3 (mesmos do v10/01).
Saída: data/fontes_a/<rid>_xr.tif (0-100, 255 = sem dado) e reports/xgb_regional.json.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
import rasterio
import torch
import xgboost as xgb
from scipy.ndimage import binary_erosion

ROOT = Path(__file__).parent
V10 = ROOT.parent / "v10"
_s = importlib.util.spec_from_file_location("f10", V10 / "01_fontes.py"); f = importlib.util.module_from_spec(_s); _s.loader.exec_module(f)
P10, OUT = V10 / "predictions/2025", ROOT / "data/fontes_a"
DEV = "cuda"
CAP_P, CAP_N = 60000, 120000  # por município


def bons():
    a = pd.DataFrame(json.loads((V10 / "reports/area_caju_2025_v10.json").read_text(encoding="utf-8")))
    a["cod"] = a.regiao.str[1:8].astype(int)
    p = pd.read_csv(ROOT.parent / "dados_externos/ranking_caju_caatinga_2023_2025.csv")[["cod", "area_ha_2025"]]
    a = a.merge(p, on="cod"); a["razao"] = a.caju_ha / a.area_ha_2025
    return a, set(a.regiao[a.razao.between(0.5, 2.0)])


def amostra(rid):
    g = torch.Generator(device=DEV).manual_seed(int(rid[1:8]))
    with rasterio.open(f.EMB / f"{rid}.vrt") as src, rasterio.open(P10 / f"{rid}_prob.tif") as pu:
        H, W, tr, crs = src.height, src.width, src.transform, src.crs
        U = pu.read(1)
        pos = (U >= 80) & (U != 255); neg = U <= 5
        if rid == "m2413359_SerradoMel":
            pos |= binary_erosion(f.polys((H, W), tr, crs))
        npos, nneg = max(pos.sum(), 1), max(neg.sum(), 1)
        qp, qn = min(1.0, 1.5 * CAP_P / npos), min(1.0, 1.5 * CAP_N / nneg)
        Xp, Xn = [], []
        for r0, c0, h, w in f.blocks(H, W):
            X, v = f.feats(src, r0, c0, h, w)
            for m, L, q in ((pos, Xp, qp), (neg, Xn, qn)):
                t = torch.from_numpy(m[r0:r0 + h, c0:c0 + w].reshape(-1)).to(DEV) & v
                idx = torch.nonzero(t).squeeze(1)
                if len(idx):
                    L.append(X[idx[torch.rand(len(idx), device=DEV, generator=g) < q]].cpu())
    cat = lambda L, c: torch.cat(L)[:c] if L else torch.empty(0, 128)
    return cat(Xp, CAP_P), cat(Xn, CAP_N)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a, ok = bons()
    rep = {"bons": sorted(ok)}
    S = {r: amostra(r) for r in f.rids() if r in ok}
    print("amostras ok:", {r: (len(p), len(n)) for r, (p, n) in S.items()}, flush=True)
    models = {}
    for reg, cods in f.REG.items():
        Xp, Xn = [], []
        for r, (p, n) in S.items():
            k = 1.0 if int(r[1:8]) in cods else 0.3
            Xp.append(p[: int(len(p) * k)]); Xn.append(n[: int(len(n) * k)])
        Xp, Xn = torch.cat(Xp), torch.cat(Xn)
        X = torch.cat([Xp, Xn]).numpy(); y = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
        clf = xgb.XGBClassifier(device="cuda", tree_method="hist", n_estimators=400, max_depth=8,
                                learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
                                scale_pos_weight=len(Xn) / max(len(Xp), 1))
        clf.fit(X, y)
        b = clf.get_booster(); b.set_param({"device": "cuda"}); b.save_model(str(ROOT / f"models/xgb_{reg}.json"))
        models[reg] = b
        rep[reg] = {"pos": len(Xp), "neg": len(Xn), "bons_da_regiao": [r for r in S if int(r[1:8]) in cods]}
        print(reg, rep[reg], flush=True)
    for rid in f.rids():
        b = models[f.REG_OF[int(rid[1:8])]]
        with rasterio.open(f.EMB / f"{rid}.vrt") as src, rasterio.open(P10 / f"{rid}_prob.tif") as pu:
            H, W = src.height, src.width; U = pu.read(1)
            XR = np.full((H, W), 255, "uint8")
            for r0, c0, h, w in f.blocks(H, W):
                X, v = f.feats(src, r0, c0, h, w)
                s = torch.from_dlpack(b.inplace_predict(cp.from_dlpack(X.contiguous()))).float()
                s = torch.where(v, (s.clamp(0, 1) * 100).round(), torch.full_like(s, 255))
                XR[r0:r0 + h, c0:c0 + w] = s.reshape(h, w).byte().cpu().numpy()
            XR[U == 255] = 255  # recorte do município igual ao mapa v10
            prof = dict(driver="GTiff", height=H, width=W, count=1, dtype="uint8", crs=src.crs, transform=src.transform,
                        nodata=255, tiled=True, blockxsize=512, blockysize=512, compress="deflate")
        with rasterio.open(OUT / f"{rid}_xr.tif", "w", **prof) as d:
            d.write(XR, 1)
        rep.setdefault("xr_ge50_ha", {})[rid] = round(float(((XR >= 50) & (XR != 255)).sum()) / 100, 1)
        print(rid, "xr>=50 ha:", rep["xr_ge50_ha"][rid], flush=True)
    (ROOT / "reports/xgb_regional.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print("XGB_REGIONAL_OK")


if __name__ == "__main__":
    main()
