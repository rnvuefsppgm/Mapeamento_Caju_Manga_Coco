"""03 — Regiões de exportação dos embeddings 2025 (GeoJSON Polygon puro, buffer 1 km) + fuso UTM.

Uso: python 03_regioes_export.py <lista>   (lista = chave em config `listas_municipios`)
Saídas: v1/data/aoi/regions/<id>.geojson, v1/data/aoi/regions_<lista>.txt ("id EPSG")
Reaproveita a expressão EE do caju (mesma imagem global, uint8 = round(x·127)+128).
"""
import json
import shutil
import sys
import unicodedata

import geopandas as gpd
from shapely.geometry import mapping

from common import cfg, ver


def slug(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s.lower().replace(" - ", "_").replace(" ", "_").replace("'", "")


c = cfg()
lista = sys.argv[1] if len(sys.argv) > 1 else "amostragem_inicial"
cods = [str(x) for x in c["listas_municipios"][lista]]
m = gpd.read_file(c["externos"]["municipios_shp"])
col = next(k for k in m.columns if k.upper().startswith("CD_MUN"))
m = m[m[col].astype(str).isin(cods)].copy()
assert len(m) == len(cods), f"achei {len(m)} de {len(cods)} municípios"

rd = ver(c, "data", "aoi", "regions", "x").parent
rd.mkdir(parents=True, exist_ok=True)
linhas = []
for _, r in m.iterrows():
    lon = r.geometry.centroid.x if m.crs.is_geographic else gpd.GeoSeries([r.geometry], crs=m.crs).to_crs(4326).centroid.x.iloc[0]
    epsg = 32700 + int((lon + 180) // 6) + 1          # UTM sul
    g = gpd.GeoSeries([r.geometry], crs=m.crs).to_crs(epsg).buffer(c["recorte"]["buffer_m"]).iloc[0]
    if g.geom_type != "Polygon":                      # DatensEE só aceita Polygon puro
        g = g.convex_hull
    g4326 = gpd.GeoSeries([g], crs=epsg).to_crs(4326).iloc[0]
    rid = f"{r[col]}_{slug(r[next(k for k in m.columns if k.upper().startswith('NM_MUN'))])}"
    (rd / f"{rid}.geojson").write_text(json.dumps(mapping(g4326)), encoding="utf-8")
    linhas.append(f"{rid} EPSG:{epsg}")
(rd.parent / f"regions_{lista}.txt").write_text("\n".join(linhas) + "\n", encoding="utf-8")

ed = ver(c, "expressions", "x").parent
ed.mkdir(parents=True, exist_ok=True)
for f in ["embeddings_2025.json", "embeddings_2025_bands.json"]:
    if not (ed / f).exists():
        shutil.copy(f"{c['embeddings']['expressao_caju']}/{f}", ed / f)
print(f"{len(linhas)} regiões ({lista}):", *linhas, sep="\n  ")
