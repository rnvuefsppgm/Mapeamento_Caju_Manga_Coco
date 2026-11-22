"""07 — Folhas PNG de alta resolução (Esri World Imagery, z17 ≈ 1,2 m/px) para interpretar a copa.

Lê <lista>_reps.csv do 06 (e o gpkg de manchas para o contorno). 6 painéis por folha (3×2, 400 px
≈ 480 m), contorno da mancha em amarelo, rótulo com id/região/área/classe MapBiomas.
Critérios de copa: coco = estrelas (folhas pinadas radiais) em grade regular ~7,5–9 m, copas
isoladas; manga = copa redonda, densa, verde-escura, 8–12 m; uva = linhas de latada contínuas;
banana = textura fina e desordenada; caatinga = copas irregulares sem grade.
Saída: v1/data/rotulos/grupos/<lista>_folha_<n>.png  |  Uso: python 07_folhas_visuais.py <lista>
"""
import io
import math
import sys

import geopandas as gpd
import pandas as pd
import requests
from PIL import Image, ImageDraw

from common import cfg, ext, ver

c = cfg()
lista = sys.argv[1]
Z = int(sys.argv[2]) if len(sys.argv) > 2 else 17
P = 400
IDS = sys.argv[3].split(",") if len(sys.argv) > 3 else None
d = ver(c, "data", "rotulos", "grupos", "x").parent
reps = pd.read_csv(d / f"{lista}_reps.csv")
if IDS:
    reps = reps[reps.id.isin(IDS)]
man = gpd.read_file(d / f"{lista}_manchas.gpkg")
cache = ext(c, "esri_cache", "x").parent
cache.mkdir(parents=True, exist_ok=True)
URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ses = requests.Session()
ses.headers["User-Agent"] = "mapa-coco-visual/1.0"


def px(lon, lat):
    n = 2 ** Z
    x = (lon + 180) / 360 * n * 256
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n * 256
    return x, y


def tile(x, y):
    f = cache / f"{Z}_{x}_{y}.jpg"
    if not f.exists():
        r = ses.get(URL.format(z=Z, x=x, y=y), timeout=60)
        r.raise_for_status()
        f.write_bytes(r.content)
    return Image.open(io.BytesIO(f.read_bytes())).convert("RGB")


def painel(r):
    cx, cy = px(r.lon, r.lat)
    x0, y0 = int(cx - P / 2), int(cy - P / 2)
    im = Image.new("RGB", (P, P))
    for tx in range(x0 // 256, (x0 + P) // 256 + 1):
        for ty in range(y0 // 256, (y0 + P) // 256 + 1):
            im.paste(tile(tx, ty), (tx * 256 - x0, ty * 256 - y0))
    dr = ImageDraw.Draw(im)
    g = man[(man.regiao == r.regiao) & (man.geometry.contains(gpd.points_from_xy([r.lon], [r.lat])[0]))]
    for geom in g.geometry:
        for poly in getattr(geom, "geoms", [geom]):
            dr.line([(px(*p)[0] - x0, px(*p)[1] - y0) for p in poly.exterior.coords], fill=(255, 255, 0), width=1)
    dr.rectangle([0, 0, P, 14], fill=(0, 0, 0))
    dr.text((3, 1), f"{r.id} {r.regiao.split('_', 1)[1]} {r.area_ha}ha mb{r.mb} g{r.grupo}", fill=(255, 255, 255))
    return im


for i in range(0, len(reps), 6):
    folha = Image.new("RGB", (3 * P + 8, 2 * P + 4), (20, 20, 20))
    for j, (_, r) in enumerate(reps.iloc[i:i + 6].iterrows()):
        folha.paste(painel(r), ((j % 3) * (P + 4), (j // 3) * (P + 4)))
    f = d / f"{lista}_folha{'_z' + str(Z) if IDS else ''}_{i // 6 + 1}.png"
    folha.save(f)
    print(f)
