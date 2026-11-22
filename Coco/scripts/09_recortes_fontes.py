"""09 — Fontes na grade de 10 m de cada região (união dos tiles de embeddings): MapBiomas 2025,
coqueiro Descals 2020 (0/1) e probabilidade de caju v10 (0–100; 255 = sem dado). Retomável.
Saídas: v1/data/fontes/<regiao>_{mb,descals,caju}.tif (uint8, GTiff tiled deflate)
Uso: python 09_recortes_fontes.py <lista>
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds

from common import cfg, ext, ver

c = cfg()
lista = sys.argv[1]
emb = ver(c, "data", "embeddings", "2025", "x").parent
out = ver(c, "data", "fontes", "x").parent
out.mkdir(parents=True, exist_ok=True)
regioes = [l.split() for l in ver(c, "data", "aoi", f"regions_{lista}.txt").read_text().split("\n") if l.strip()]

# índice de bounds das fontes multi-arquivo
def indice(arqs, nome):
    f = ext(c, "indices", f"{nome}.json")
    if f.exists():
        return json.loads(f.read_text())
    idx = {}
    for a in arqs:
        with rasterio.open(a) as s:
            idx[str(a)] = [list(transform_bounds(s.crs, "EPSG:4326", *s.bounds)), s.crs.to_string()]
    f.write_text(json.dumps(idx))
    return idx

desc = indice(sorted(Path(ext(c, "descals_coco", "raw", "x").parent).glob("GCL_2020_*.tif")), "descals")
caju = indice(sorted(Path(c["externos"]["caju_pred_dir"].replace("v10b", "v10")).glob("*_prob.tif")), "caju_v10")


def grade(rid, crs):
    bs = []
    for t in (emb / rid).glob("*.tif"):
        with rasterio.open(t) as s:
            bs.append(s.bounds)
    x0, y0 = min(b.left for b in bs), min(b.bottom for b in bs)
    x1, y1 = max(b.right for b in bs), max(b.top for b in bs)
    w, h = int(round((x1 - x0) / 10)), int(round((y1 - y0) / 10))
    return from_origin(x0, y1, 10, 10), (h, w), crs


def mosaico(idx, tr, shp, crs, fill, resamp, combina):
    dst = np.full(shp, fill, np.float32)
    b = transform_bounds(crs, "EPSG:4326", tr.c, tr.f + tr.e * shp[0], tr.c + tr.a * shp[1], tr.f)
    for a, (bb, _) in idx.items():
        if bb[0] > b[2] or bb[2] < b[0] or bb[1] > b[3] or bb[3] < b[1]:
            continue
        tmp = np.full(shp, np.nan, np.float32)
        with rasterio.open(a) as s:
            reproject(rasterio.band(s, 1), tmp, dst_transform=tr, dst_crs=crs, resampling=resamp,
                      src_nodata=s.nodata, dst_nodata=np.nan)
        ok = ~np.isnan(tmp)
        dst[ok] = combina(np.where(dst[ok] == fill, -1, dst[ok]), tmp[ok]) if fill != 0 else np.maximum(dst[ok], tmp[ok])
    return dst


def grava(arr, tr, crs, f):
    with rasterio.open(f, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype="uint8",
                       crs=crs, transform=tr, tiled=True, compress="deflate") as d:
        d.write(arr.astype(np.uint8), 1)


for rid, crs in regioes:
    tr, shp, crs = grade(rid, crs)
    fs = {k: out / f"{rid}_{k}.tif" for k in ["mb", "descals", "caju"]}
    if all(f.exists() for f in fs.values()):
        continue
    with rasterio.open(c["externos"]["mapbiomas_2025"]) as m:
        mb = np.zeros(shp, np.uint8)
        reproject(rasterio.band(m, 1), mb, dst_transform=tr, dst_crs=crs, resampling=Resampling.nearest)
    grava(mb, tr, crs, fs["mb"])
    d = mosaico(desc, tr, shp, crs, 0, Resampling.nearest, np.maximum)
    grava(np.nan_to_num(d) > 0, tr, crs, fs["descals"])
    cj = mosaico(caju, tr, shp, crs, 255, Resampling.bilinear, np.maximum)
    grava(np.clip(cj, 0, 255), tr, crs, fs["caju"])
    print(f"{rid}: {shp[1]}x{shp[0]} px | descals {100 * (d > 0).mean():.2f}% "
          f"({(d > 0).sum() / 100:,.0f} ha) | caju c/ dado {100 * (cj != 255).mean():.0f}% | "
          f"mb48 {(mb == 48).sum() / 100:,.0f} ha", flush=True)
