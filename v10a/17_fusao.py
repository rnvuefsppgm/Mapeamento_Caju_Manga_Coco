"""17 (v10a-ft) — Escolhe a fusão U-Net × XGBoost regional SEM vazamento: mapa fora do fold de Serra do Mel
(cada bloco de teste previsto só pelo modelo do fold que não o viu, com TTA) contra os rótulos de campo
(polígonos = 1; fora da faixa de 25 m = 0), só na área dos chips de campo. Testa a = 1,0 ... 0,5
em p = a·U + (1-a)·XGB e o limiar 0,30-0,70. Saída: reports/fusao.json
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.features import rasterize
from rasterio.warp import transform_bounds

from src import pred_a as P

ROOT = Path(__file__).parent
SM = "m2413359_SerradoMel"
VRT = ROOT.parent / f"v8/data/ee_exports/2025/embeddings/{SM}.vrt"


def iou(p, t, thr):
    c = p >= thr; tp = (c & (t == 1)).sum(); fp = (c & (t == 0)).sum(); fn = (~c & (t == 1)).sum()
    return tp / max(tp + fp + fn, 1)


def main():
    folds = sorted((ROOT / "models/ensemble_unet_conc_a").glob("fold*.pt"))
    with rasterio.open(VRT) as s:
        H, W, tr, crs = s.height, s.width, s.transform, s.crs
    bx = np.floor((tr.c + (np.arange(W) + 0.5) * tr.a) / 10000).astype(np.int64)
    by = np.floor((tr.f + (np.arange(H) + 0.5) * tr.e) / 10000).astype(np.int64)
    blk = bx[None, :] * 100000 + by[:, None]
    U = np.full((H, W), np.nan, "float32")
    for f in folds:
        tb = torch.load(f, map_location="cpu", weights_only=False)["test_blocks"]
        ids = [int(b.split("_")[1]) * 100000 + int(b.split("_")[2]) for b in tb]
        msk = np.isin(blk, ids)
        p, _ = P.predict(VRT, P.load_members([f]))
        U[msk] = p[msk]; print(f.name, "ok", flush=True)
    XR = rasterio.open(ROOT / f"data/fontes_a/{SM}_xr.tif").read(1).astype("float32")
    XR[XR == 255] = np.nan; XR /= 100
    g = gpd.read_file(ROOT.parent / "v7/data/vectors/Pol_Caju.shp").to_crs(crs)
    pol = rasterize([(x, 1) for x in g.geometry if x is not None], out_shape=(H, W), transform=tr, fill=0).astype(bool)
    ring = rasterize([(x, 1) for x in g.buffer(25).geometry if x is not None], out_shape=(H, W), transform=tr, fill=0).astype(bool)
    area = np.zeros((H, W), bool)  # área dos chips de campo (tiles v7)
    for e in (ROOT.parent / "v7/data/embeddings").glob("emb_features_*_2025.tif"):
        with rasterio.open(e) as d:
            b = transform_bounds(d.crs, crs, *d.bounds)
        r0, c0 = rasterio.transform.rowcol(tr, b[0], b[3]); r1, c1 = rasterio.transform.rowcol(tr, b[2], b[1])
        area[max(r0, 0):max(r1, 0), max(c0, 0):max(c1, 0)] = True
    t = np.full((H, W), 255, "uint8"); t[~ring] = 0; t[pol] = 1
    ok = area & np.isfinite(U) & np.isfinite(XR) & (t != 255)
    u, x, tt = U[ok], XR[ok], t[ok]
    res = []
    for a in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        p = a * u + (1 - a) * x
        best = max(((iou(p, tt, th), th) for th in np.arange(0.30, 0.71, 0.05)), key=lambda z: z[0])
        res.append({"a": a, "iou": round(float(best[0]), 4), "limiar": round(float(best[1]), 2)})
        print(res[-1], flush=True)
    b = max(res, key=lambda r: r["iou"]); u1 = res[0]
    out = {"testes": res, "escolha": b if b["iou"] >= u1["iou"] + 0.005 else u1, "pixels": int(ok.sum())}
    (ROOT / "reports/fusao.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("FUSAO_OK", out["escolha"])


if __name__ == "__main__":
    main()
