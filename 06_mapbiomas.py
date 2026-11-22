"""06 — MapBiomas Col. 11, 2025 (30 m) para manga (porte do caju v10b/30_mapbiomas_recorte.py).

--classes : classes MapBiomas dentro dos polígonos de manga (amostra a 10 m) e no município todo de Livramento
            -> reports/mb_classes_manga.csv (define o conjunto C de classes compatíveis com manga, como no caju)
(padrão)  : recorta a MapBiomas para a grade de 10 m do VRT de cada região (vizinho mais próximo)
            -> dados/mapbiomas/<rid>_mb.tif ; e área por classe por município -> reports/mb_area_municipios.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from src.config import load_config, resolve_path, setup_logging

cfg = load_config(Path(__file__).parent / "config/config.yaml")
log = setup_logging("06_mapbiomas", cfg)
M = cfg["manga"]
MB = M["fontes"]["mapbiomas_tif"]
PROF = dict(driver="GTiff", count=1, dtype="uint8", nodata=0, tiled=True, blockxsize=512, blockysize=512, compress="deflate")


def reproj(m, H, W, tr, crs):
    out = np.zeros((H, W), "uint8")
    reproject(rasterio.band(m, 1), out, dst_transform=tr, dst_crs=crs, dst_nodata=0, resampling=Resampling.nearest)
    return out


def classes():
    p = gpd.read_file(resolve_path(cfg, "amostras") / "manga_poligonos.gpkg").to_crs(M["crs"])
    mun = gpd.read_file(resolve_path(cfg, "pam") / "municipios_selecionados.gpkg").to_crs(M["crs"])
    liv = mun[mun.cod.isin(M["selecao"]["incluir_sempre"])]
    x0, y0, x1, y1 = liv.total_bounds
    x0, y1 = np.floor(x0 / 10) * 10, np.ceil(y1 / 10) * 10
    W, H = int(np.ceil((x1 - x0) / 10)), int(np.ceil((y1 - y0) / 10))
    tr = from_origin(x0, y1, 10, 10)
    with rasterio.open(MB) as m:
        cl = reproj(m, H, W, tr, M["crs"])
    poly = rasterize(((g, 1) for g in p.geometry), (H, W), transform=tr, dtype="uint8").astype(bool)
    munm = rasterize(((g, 1) for g in liv.geometry), (H, W), transform=tr, dtype="uint8").astype(bool)
    rows = []
    for k, s in {"poligonos_manga": poly, "municipio_livramento": munm}.items():
        v, c = np.unique(cl[s], return_counts=True)
        for a, b in zip(v, c):
            rows.append({"sel": k, "classe": int(a), "ha": b / 100})
    df = pd.DataFrame(rows)
    df["pct"] = 100 * df.ha / df.groupby("sel").ha.transform("sum")
    df.to_csv(resolve_path(cfg, "reports") / "mb_classes_manga.csv", index=False)
    t = df[df.sel == "poligonos_manga"].sort_values("pct", ascending=False)
    log.info("RESUMO classes nos polígonos: " + "; ".join(f"{r.classe}={r.pct:.1f}%" for r in t.head(8).itertuples()))


def recorte():
    emb, out = resolve_path(cfg, "embeddings"), resolve_path(cfg, "mapbiomas")
    rids = [r for r in (resolve_path(cfg, "aoi") / "regioes.txt").read_text().split() if r]
    rows = []
    with rasterio.open(MB) as m:
        for rid in rids:
            v = emb / f"{rid}.vrt"
            if not v.exists():
                log.info(f"{rid}: sem VRT, pulado")
                continue
            with rasterio.open(v) as e:
                H, W, tr, crs = e.height, e.width, e.transform, e.crs
            cl = reproj(m, H, W, tr, crs)
            with rasterio.open(out / f"{rid}_mb.tif", "w", height=H, width=W, crs=crs, transform=tr, **PROF) as d:
                d.write(cl, 1)
            vv, cc = np.unique(cl, return_counts=True)
            rows += [{"regiao": rid, "classe": int(a), "ha": b / 100} for a, b in zip(vv, cc) if a]
    pd.DataFrame(rows).to_csv(resolve_path(cfg, "reports") / "mb_area_municipios.csv", index=False)
    log.info(f"RESUMO: {len({r['regiao'] for r in rows})}/{len(rids)} recortes")


if __name__ == "__main__":
    classes() if "--classes" in sys.argv else recorte()
