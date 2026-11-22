"""19 (v1) — Mapas de manga em VETOR para abrir no Google Earth / GIS: por município e total dos 25.
Polígonos das manchas de manga (classe 1) de dados/predicoes/v1/<rid>_class.tif, simplificados (5 m), com área (ha)
e probabilidade média. Saídas: dados/predicoes/v1/vetores/<rid>_manga.{gpkg,kmz} e manga_2025_25mun.{gpkg,kmz}.
"""
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape

R = Path(__file__).parent; V1 = R / "dados/predicoes/v1"; OUT = V1 / "vetores"; OUT.mkdir(exist_ok=True)
pam = pd.read_csv(R / "dados/pam/ranking_pam_censo.csv").set_index("cod")["nome"]


def kmz(g, path, nome):
    L = ['<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>', f"<name>{nome}</name>",
         '<Style id="m"><LineStyle><color>ff00a5ff</color><width>1.5</width></LineStyle><PolyStyle><color>6600a5ff</color></PolyStyle></Style>']
    for mun, sub in g.groupby("municipio"):
        L.append(f"<Folder><name>{mun} ({sub.area_ha.sum():,.0f} ha)</name>")
        for r in sub.itertuples():
            gs = [r.geometry] if r.geometry.geom_type == "Polygon" else list(r.geometry.geoms)
            geo = "".join("<Polygon><outerBoundaryIs><LinearRing><coordinates>"
                          + " ".join(f"{x:.6f},{y:.6f}" for x, y in p.exterior.coords)
                          + "</coordinates></LinearRing></outerBoundaryIs></Polygon>" for p in gs)
            L.append(f"<Placemark><name>{r.area_ha:.1f} ha</name><styleUrl>#m</styleUrl><description>p média {r.p_media:.2f}"
                     f"</description><MultiGeometry>{geo}</MultiGeometry></Placemark>")
        L.append("</Folder>")
    L.append("</Document></kml>")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", "\n".join(L))


todos = []
for f in sorted(V1.glob("m*_class.tif")):
    rid = f.name.replace("_class.tif", ""); cod = int(rid[1:8])
    with rasterio.open(f) as d:
        C = d.read(1) == 1; tr, crs = d.transform, d.crs
    with rasterio.open(V1 / f"{rid}_prob.tif") as d:
        P = d.read(1).astype("float32")
    lab, n = ndimage.label(C)
    if not n:
        continue
    pm = ndimage.mean(P, lab, range(1, n + 1)) / 100
    rows = [{"lab": int(v), "geometry": shape(g)} for g, v in shapes(lab.astype(np.int32), mask=C, transform=tr)]
    g = gpd.GeoDataFrame(rows, crs=crs).dissolve("lab").reset_index()
    g["area_ha"] = g.area / 1e4
    g["p_media"] = pm[g.lab.values - 1]
    g["geometry"] = g.simplify(5)
    g["municipio"], g["cod"] = pam.get(cod, rid), cod
    g = g.drop(columns="lab").to_crs(4326)
    g.to_file(OUT / f"{rid}_manga.gpkg", driver="GPKG")
    kmz(g, OUT / f"{rid}_manga.kmz", f"Manga 2025 v1 — {pam.get(cod, rid)}")
    todos.append(g)
    print(f"{rid}: {len(g)} manchas, {g.area_ha.sum():,.0f} ha", flush=True)
T = gpd.GeoDataFrame(pd.concat(todos, ignore_index=True), crs=4326)
T.to_file(OUT / "manga_2025_25mun.gpkg", driver="GPKG")
kmz(T, OUT / "manga_2025_25mun.kmz", "Manga 2025 v1 — 25 municípios")
print(f"VETORES_OK: {len(T)} manchas, {T.area_ha.sum():,.0f} ha em {T.municipio.nunique()} municípios -> {OUT}")
