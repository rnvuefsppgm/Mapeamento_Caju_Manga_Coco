"""15 — QC dos rótulos automáticos: pontos aleatórios estratificados → folhas de 18 painéis (Esri z17, 200 px ≈ 240 m,
mira no centro) para conferência visual. Decisões em v1/data/rotulos/qc/qc_rotulos_decisoes.csv (id, classe_visual).
Uso: python 15_qc_rotulos.py <lista> [n_coco_por_macro=36] [n_nao_por_macro=18]
"""
import io
import math
import sys

import numpy as np
import pandas as pd
import rasterio
import requests
from PIL import Image, ImageDraw
from pyproj import Transformer

from common import cfg, ext, ver

c = cfg()
lista = sys.argv[1]
NC = int(sys.argv[2]) if len(sys.argv) > 2 else 36
NN = int(sys.argv[3]) if len(sys.argv) > 3 else 18
RD = sys.argv[4] if len(sys.argv) > 4 else "r2"   # rodada do QC
sf = {str(x) for x in c["listas_municipios"]["sao_francisco"]}
rot = ver(c, "data", "rotulos", "x").parent
qd = rot / "qc"
regioes = [l.split() for l in ver(c, "data", "aoi", f"regions_{lista}.txt").read_text().split("\n") if l.strip()]
rng = np.random.default_rng(sum(map(ord, RD)))   # determinístico por rodada
pts = []
for rid, crs in regioes:
    with rasterio.open(rot / "auto" / f"{rid}_rotulo.tif") as s:
        L, tr = s.read(1, out_shape=(s.height // 4, s.width // 4)), s.transform * s.transform.scale(4, 4)
    tf = Transformer.from_crs(crs, 4326, always_xy=True)
    for v in [0, 1]:
        rr, cc = np.nonzero(L == v)
        k = min(200, rr.size)
        for i in rng.choice(rr.size, k, replace=False) if k else []:
            x, y = tr * (cc[i] + 0.5, rr[i] + 0.5)
            pts.append((rid, "sf" if rid.split("_")[0] in sf else "litoral", v, *tf.transform(x, y)))
p = pd.DataFrame(pts, columns=["regiao", "macro", "rotulo", "lon", "lat"])
s = pd.concat([g.sample(min(NC if r == 1 else NN, len(g)), random_state=len(RD) * 7) for (m, r), g in p.groupby(["macro", "rotulo"])])
s = s.sample(frac=1, random_state=5).reset_index(drop=True)   # embaralha: rótulo não aparece na folha
s["id"] = [f"q{i:03d}" for i in range(len(s))]
s.to_csv(qd / f"qc_rotulos_pontos_{RD}.csv", index=False)

Z, P = 17, 200
cache = ext(c, "esri_cache", "x").parent
URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ses = requests.Session(); ses.headers["User-Agent"] = "mapa-coco-visual/1.0"


def tile(x, y):
    f = cache / f"{Z}_{x}_{y}.jpg"
    if not f.exists():
        f.write_bytes(ses.get(URL.format(z=Z, x=x, y=y), timeout=60).content)
    return Image.open(io.BytesIO(f.read_bytes())).convert("RGB")


def painel(r):
    n = 2 ** Z
    cx = (r.lon + 180) / 360 * n * 256
    cy = (1 - math.asinh(math.tan(math.radians(r.lat))) / math.pi) / 2 * n * 256
    x0, y0 = int(cx - P / 2), int(cy - P / 2)
    im = Image.new("RGB", (P, P))
    for tx in range(x0 // 256, (x0 + P) // 256 + 1):
        for ty in range(y0 // 256, (y0 + P) // 256 + 1):
            im.paste(tile(tx, ty), (tx * 256 - x0, ty * 256 - y0))
    d = ImageDraw.Draw(im)
    d.line([(P / 2 - 12, P / 2), (P / 2 - 4, P / 2)], fill=(255, 0, 0), width=2)
    d.line([(P / 2 + 4, P / 2), (P / 2 + 12, P / 2)], fill=(255, 0, 0), width=2)
    d.line([(P / 2, P / 2 - 12), (P / 2, P / 2 - 4)], fill=(255, 0, 0), width=2)
    d.line([(P / 2, P / 2 + 4), (P / 2, P / 2 + 12)], fill=(255, 0, 0), width=2)
    d.rectangle([0, 0, 34, 12], fill=(0, 0, 0)); d.text((2, 0), r.id, fill=(255, 255, 255))
    return im


for i in range(0, len(s), 18):
    f = Image.new("RGB", (6 * (P + 2), 3 * (P + 2)), (20, 20, 20))
    for j, (_, r) in enumerate(s.iloc[i:i + 18].iterrows()):
        f.paste(painel(r), ((j % 6) * (P + 2), (j // 6) * (P + 2)))
    f.save(qd / f"qc_rotulos_{RD}_folha_{i // 18 + 1}.png")
print(f"{len(s)} pontos ({s.rotulo.sum()} coco) em {(len(s) + 17) // 18} folhas -> {qd}")
