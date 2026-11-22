"""12 — Rótulos verificados → v1/data/rotulos/r_ouro.gpkg (EPSG:4326).

nivel=ouro : polígonos do usuário (amostras_usuario/*.km[lz], pasta COCO=coco, NAO_COCO=nao_coco) e
             decisões visuais do QC (qc/qc_decisoes.csv; mancha ∩ raio de 150 m do ponto conferido).
nivel=prata: modelos de grupo confirmados pela copa (grupos/modelo_*.kml) — só treino.
'duvida' fica fora. Uso: python 12_importar_rotulos.py
"""
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon

from common import cfg, ver

c = cfg()
d = ver(c, "data", "rotulos", "x").parent
UTM = c["project"]["crs_working"]


def kml_polys(p, pasta_padrao=None):
    p = Path(p)
    raw = zipfile.ZipFile(p).read(next(n for n in zipfile.ZipFile(p).namelist() if n.endswith(".kml"))) \
        if p.suffix.lower() == ".kmz" else p.read_bytes()
    root = ET.fromstring(raw)
    for el in root.iter():
        el.tag = re.sub(r"\{.*\}", "", el.tag)
    out = []
    for fo in root.iter("Folder"):
        nome = (fo.findtext("name") or "").strip().upper()
        cl = {"COCO": "coco", "COCO_OK": "coco", "NAO_COCO": "nao_coco"}.get(nome, pasta_padrao)
        if not cl:
            continue
        for pm in fo.findall("Placemark"):
            for pg in pm.iter("Polygon"):
                xy = [tuple(map(float, t.split(",")[:2])) for t in pg.find(".//outerBoundaryIs//coordinates").text.split()]
                out.append({"classe": cl, "geometry": Polygon(xy), "origem": f"{p.name}:{pm.findtext('name')}"})
    return out


reg = []
for f in sorted(Path(c["amostras_usuario"]["kml_dir"]).glob("*.km[lz]")):
    reg += [dict(r, nivel="ouro", fonte="usuario") for r in kml_polys(f)]
for f in sorted((d / "grupos").glob("modelo_*.kml")):
    reg += [dict(r, nivel="prata", fonte="grupo_visual") for r in kml_polys(f, "coco")]

q = pd.read_csv(d / "qc" / "qc_decisoes.csv")
q = q[q.classe.isin(["coco", "nao_coco"])]
for tag in sorted(set(q.id_qc.str.split("_").str[0])):
    reps = pd.read_csv(d / "grupos" / f"{tag}_reps.csv").merge(q, left_on="id", right_on="id_qc")
    man = gpd.read_file(d / "grupos" / f"{tag}_manchas.gpkg").to_crs(UTM)
    for _, r in reps.iterrows():
        pt = gpd.GeoSeries([Point(r.lon, r.lat)], crs=4326).to_crs(UTM).iloc[0]
        m = man[man.contains(pt)]
        g = (m.union_all() if len(m) else pt.buffer(50)).intersection(pt.buffer(150))
        reg.append({"classe": r.classe, "geometry": gpd.GeoSeries([g], crs=UTM).to_crs(4326).iloc[0],
                    "origem": f"{r.id_qc}:{r.nota}", "nivel": "ouro", "fonte": "qc_visual", "regiao": r.regiao})

for qp in sorted((d / "qc").glob("qc_rotulos_decisoes_r*.csv")):   # QC cego (15): pontos conferidos viram ouro (15 m)
    for _, r in pd.read_csv(qp).query("visual != 'duvida'").iterrows():
        g = gpd.GeoSeries([Point(r.lon, r.lat)], crs=4326).to_crs(UTM).buffer(15).to_crs(4326).iloc[0]
        reg.append({"classe": r.visual, "geometry": g, "origem": f"qc_rotulos:{r.id}", "nivel": "ouro",
                    "fonte": "qc_pontos", "regiao": r.regiao})
g = gpd.GeoDataFrame(reg, crs=4326)
g = g[~g.is_empty & g.is_valid]
g.to_file(d / "r_ouro.gpkg", driver="GPKG")
a = g.to_crs(UTM).area / 1e4
print(g.assign(ha=a).groupby(["nivel", "classe", "fonte"]).agg(n=("ha", "size"), ha=("ha", "sum")).round(1).to_string())
