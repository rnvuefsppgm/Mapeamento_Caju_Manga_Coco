"""41 — Raster ÚNICO dos municípios (EPSG:32724, 10 m): classe (0/1; 255 = fora) e probabilidade (0–100; 255 = fora).
Regiões em outro fuso (ex.: Touros, 32725) são reprojetadas por vizinho mais próximo.
Saídas: v1/predictions/coco_2025_<lista>_classe.tif e _prob.tif (GTiff tiled deflate + overviews)
Uso: python 41_mosaico.py <lista>
"""
import sys

import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT

from common import cfg, ver

c = cfg()
lista = sys.argv[1]
CRS = c["project"]["crs_working"]
pdir = ver(c, "predictions", "teste" if lista == "teste" else "2025", "x").parent
regioes = [l.split()[0] for l in ver(c, "data", "aoi", f"regions_{lista}.txt").read_text().split("\n") if l.strip()]
for k in ["classe", "prob"]:
    srcs = [rasterio.open(pdir / f"{r}_{k}.tif") for r in regioes]
    vrts = [s if s.crs.to_string() == CRS else WarpedVRT(s, crs=CRS, resampling=Resampling.nearest, nodata=255) for s in srcs]
    a, tr = merge(vrts, res=10, nodata=255, method="min" if k == "classe" else "max")
    # 'min' na classe: onde municípios vizinhos se sobrepõem (buffer), 0/1 vence 255; 'max' na prob só onde ambos válidos
    prof = dict(driver="GTiff", height=a.shape[1], width=a.shape[2], count=1, dtype="uint8", crs=CRS, transform=tr,
                nodata=255, tiled=True, compress="deflate", BIGTIFF="IF_SAFER")
    f = ver(c, "predictions", f"coco_2025_{lista}_{k}.tif")
    if k == "prob":
        a[0][a[0] == 255] = 255
    with rasterio.open(f, "w", **prof) as d:
        d.write(a[0], 1)
        d.build_overviews([2, 4, 8, 16, 32], Resampling.nearest)
    for s in srcs:
        s.close()
    print(f"{f.name}: {a.shape[2]}x{a.shape[1]} px; coco {(a[0] == 1).sum() / 100:,.0f} ha" if k == "classe" else f"{f.name} ok")
