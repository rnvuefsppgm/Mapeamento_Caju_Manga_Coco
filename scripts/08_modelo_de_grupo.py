"""08 — Modelo regional (KML, pasta COCO) a partir de grupos do 06 confirmados como coco pela copa (07).

Pega as manchas mais típicas dos grupos (menor distância ao centróide, área ≥ 3 ha, até N),
erode 15 m (tira borda mista) e grava KML para o 05 usar como modelo:
  python 05_candidatos_similaridade.py <lista> <kml>
Uso: python 08_modelo_de_grupo.py <lista> <grupos ex. 11,6> [N=40]
"""
import sys

import geopandas as gpd

from common import cfg, ver

c = cfg()
lista, grupos = sys.argv[1], [int(g) for g in sys.argv[2].split(",")]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 40
d = ver(c, "data", "rotulos", "grupos", "x").parent
m = gpd.read_file(d / f"{lista}_manchas.gpkg")
s = m[m.grupo.isin(grupos) & (m.area_ha >= 3)].sort_values("dist").head(N)
s = s.to_crs(c["project"]["crs_working"])
s["geometry"] = s.buffer(-15)
s = s[~s.is_empty].explode(index_parts=False)
s = s[s.area >= 5000].to_crs(4326)
pm = []
for i, (_, r) in enumerate(s.iterrows()):
    co = " ".join(f"{x:.6f},{y:.6f},0" for x, y in r.geometry.exterior.coords)
    pm.append(f"<Placemark><name>sf_g{r.grupo}_{i:02d}</name><Polygon><outerBoundaryIs><LinearRing>"
              f"<coordinates>{co}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>")
f = d / f"modelo_{lista}_g{'-'.join(map(str, grupos))}.kml"
f.write_text('<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
             f"<Folder><name>COCO</name>{''.join(pm)}</Folder></Document></kml>", encoding="utf-8")
print(f"{len(pm)} polígonos-modelo ({s.to_crs(c['project']['crs_working']).area.sum() / 1e4:,.0f} ha) "
      f"por região: {s.regiao.value_counts().to_dict()} -> {f}")
