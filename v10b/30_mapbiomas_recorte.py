"""30 (v10b) — Recorta a MapBiomas Col. 11, 2025 (30 m, D:/Eduarda/raster_new) para a grade de 10 m dos embeddings
de cada município (vizinho mais próximo) e mede as classes dentro dos polígonos de campo (Serra do Mel) e dos
pixels de caju confiável da v10 (>= 0,80) por região. Saídas: data/mapbiomas/<rid>_mb.tif, reports/mb_classes_caju.csv
"""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject

ROOT = Path(__file__).parent
_s = importlib.util.spec_from_file_location("f10", ROOT.parent / "v10/01_fontes.py"); f = importlib.util.module_from_spec(_s); _s.loader.exec_module(f)
MB = Path(os.environ.get("MB_RASTER", "D:/Eduarda/raster_new/brazil_coverage-col11_2025.tif"))  # MapBiomas Col. 11, 2025 (30 m)
P10 = ROOT.parent / "v10/predictions/2025"
rows = []
with rasterio.open(MB) as m:
    for rid in f.rids():
        with rasterio.open(f.EMB / f"{rid}.vrt") as e:
            H, W, tr, crs = e.height, e.width, e.transform, e.crs
        out = np.zeros((H, W), "uint8")
        reproject(rasterio.band(m, 1), out, dst_transform=tr, dst_crs=crs, dst_nodata=0, resampling=Resampling.nearest)
        prof = dict(driver="GTiff", height=H, width=W, count=1, dtype="uint8", crs=crs, transform=tr, nodata=0,
                    tiled=True, blockxsize=512, blockysize=512, compress="deflate")
        with rasterio.open(ROOT / f"data/mapbiomas/{rid}_mb.tif", "w", **prof) as d:
            d.write(out, 1)
        U = rasterio.open(P10 / f"{rid}_prob.tif").read(1)
        sel = {"todo": U != 255, "v10_caju>=0.8": (U >= 80) & (U != 255)}
        if rid == "m2413359_SerradoMel":
            sel["poligonos_campo"] = f.polys((H, W), tr, crs)
        for k, s in sel.items():
            v, c = np.unique(out[s], return_counts=True)
            for a, b in zip(v, c):
                rows.append({"regiao": rid, "reg": f.REG_OF[int(rid[1:8])], "sel": k, "classe": int(a), "ha": b / 100})
        print(rid, "ok", flush=True)
df = pd.DataFrame(rows); df.to_csv(ROOT / "reports/mb_classes_caju.csv", index=False)
print("MB_RECORTE_OK")
