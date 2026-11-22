"""42 — Mapa de coco em VETOR para Google Earth / SIG: por município e total da lista.
Polígonos das manchas de coco (classe 1) de v1/predictions/2025/<regiao>_classe.tif, simplificados (5 m), com área (ha)
e probabilidade média. Saídas: v1/predictions/vetores/<regiao>_coco.{gpkg,kmz} e coco_2025_<lista>.{gpkg,kmz} (EPSG:4326)
Uso: python 42_vetores_kmz.py <lista>
"""
import sys
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape

from common import cfg, ext, ver

c = cfg()
lista = sys.argv[1]
pdir = ver(c, "predictions", "2025", "x").parent
out = ver(c, "predictions", "vetores", "x").parent
out.mkdir(parents=True, exist_ok=True)
regioes = [l.split()[0] for l in ver(c, "data", "aoi", f"regions_{lista}.txt").read_text().split("\n") if l.strip()]
nomes = pd.read_csv(ext(c, "pam", "pam_censo_coco.csv"), dtype={"cod": str}).set_index("cod")["nome"]


def kmz(g, path, nome):
    L = ['<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>', f"<name>{nome}</name>",
         '<Style id="c"><LineStyle><color>ff00ff00</color><width>1.5</width></LineStyle><PolyStyle><color>6600ff00</color></PolyStyle></Style>']
    for mun, sub in g.groupby("municipio"):
        L.append(f"<Folder><name>{mun} ({sub.area_ha.sum():,.0f} ha)</name>")
        for r in sub.itertuples():
            gs = [r.geometry] if r.geometry.geom_type == "Polygon" else list(r.geometry.geoms)
            geo = "".join("<Polygon><outerBoundaryIs><LinearRing><coordinates>"
                          + " ".join(f"{x:.6f},{y:.6f}" for x, y in p.exterior.coords)
                          + "</coordinates></LinearRing></outerBoundaryIs></Polygon>" for p in gs)
            L.append(f"<Placemark><name>{r.area_ha:.1f} ha</name><styleUrl>#c</styleUrl><description>p média {r.p_media:.2f}"
                     f"</description><MultiGeometry>{geo}</MultiGeometry></Placemark>")
        L.append("</Folder>")
    L.append("</Document></kml>")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", "\n".join(L))


todos = []
for rid in regioes:
    cod = rid.split("_")[0]
    with rasterio.open(pdir / f"{rid}_classe.tif") as d:
        C = d.read(1) == 1; tr, crs = d.transform, d.crs
    with rasterio.open(pdir / f"{rid}_prob.tif") as d:
        P = d.read(1).astype("float32")
    lab, n = ndimage.label(C)
    if not n:
        print(f"{rid}: sem coco", flush=True); continue
    pm = ndimage.mean(P, lab, range(1, n + 1)) / 100
    rows = [{"lab": int(v), "geometry": shape(g)} for g, v in shapes(lab.astype(np.int32), mask=C, transform=tr)]
    g = gpd.GeoDataFrame(rows, crs=crs).dissolve("lab").reset_index()
    g["area_ha"] = g.area / 1e4
    g["p_media"] = pm[g.lab.values - 1]
    g["geometry"] = g.simplify(5)
    g["municipio"], g["cod"] = nomes.get(cod, rid), cod
    g = g.drop(columns="lab").to_crs(4326)
    g.to_file(out / f"{rid}_coco.gpkg", driver="GPKG")
    kmz(g, out / f"{rid}_coco.kmz", f"Coco 2025 v1 — {nomes.get(cod, rid)}")
    todos.append(g)
    print(f"{rid}: {len(g)} manchas, {g.area_ha.sum():,.0f} ha", flush=True)
T = gpd.GeoDataFrame(pd.concat(todos, ignore_index=True), crs=4326)
T.to_file(out / f"coco_2025_{lista}.gpkg", driver="GPKG")
kmz(T, out / f"coco_2025_{lista}.kmz", f"Coco 2025 v1 — {lista}")
print(f"VETORES_OK: {len(T)} manchas, {T.area_ha.sum():,.0f} ha em {T.municipio.nunique()} municípios -> {out}")
