"""Resumo compacto (≤ 15 linhas) de raster, vetor ou tabela — evita abrir arquivos grandes.

Uso: python tools/resumo.py <arquivo> [<arquivo> ...]
"""
import sys
from pathlib import Path


def raster(p):
    import numpy as np
    import rasterio
    with rasterio.open(p) as r:
        print(f"  {r.width}x{r.height} px, {r.count} bandas, {r.dtypes[0]}, {r.crs}, res={r.res}, nodata={r.nodata}")
        print(f"  bounds={tuple(round(b, 1) for b in r.bounds)}")
        f = max(1, max(r.width, r.height) // 1024)
        a = r.read(1, out_shape=(r.height // f or 1, r.width // f or 1))
        v = a[a != r.nodata] if r.nodata is not None else a.ravel()
        if v.size and v.dtype.kind in "ui" and np.unique(v).size <= 40:
            u, c = np.unique(v, return_counts=True)
            print("  classes(banda1, amostra):", dict(zip(u.tolist(), (c / c.sum()).round(3).tolist())))
        elif v.size:
            print(f"  banda1 min={v.min():.3g} p50={np.median(v):.3g} max={v.max():.3g}")


def vetor(p):
    import geopandas as gpd
    g = gpd.read_file(p)
    print(f"  {len(g)} feições, {g.crs}, tipos={g.geom_type.value_counts().to_dict()}")
    print(f"  colunas={list(g.columns)[:15]}")
    print(f"  bounds={tuple(round(b, 3) for b in g.total_bounds)}")


def tabela(p):
    import pandas as pd
    d = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    print(f"  {d.shape[0]} linhas x {d.shape[1]} colunas: {list(d.columns)[:15]}")
    print(d.head(5).to_string(max_colwidth=25))


for a in sys.argv[1:]:
    p = Path(a)
    print(f"# {p.name} ({p.stat().st_size / 1e6:.1f} MB)")
    s = p.suffix.lower()
    if s in {".tif", ".tiff", ".vrt"}:
        raster(p)
    elif s in {".gpkg", ".shp", ".geojson", ".kml"}:
        vetor(p)
    elif s in {".csv", ".parquet"}:
        tabela(p)
    else:
        print("  tipo não suportado")
