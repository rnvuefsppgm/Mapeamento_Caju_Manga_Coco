"""06 — Grupos de manchas de lavoura (MapBiomas 48/41) pelos embeddings, para regiões SEM amostras-modelo.

Para cada mancha ≥ 2 ha de classe 48/41: embedding médio (normalizado). K-means (K grupos) nas manchas.
Representantes (2 mais próximos do centróide, área ≥ 3 ha) vão para um visualizador de alta resolução
(Esri World Imagery, z18) onde a copa é interpretada (coco = estrela + grade ~8 m; manga = copa
redonda escura; uva = latada em linhas). Grupos marcados como coco viram modelo regional no 05.
Saídas: v1/data/rotulos/grupos/{<lista>_manchas.gpkg, <lista>_reps.csv, <lista>_visual_<n>.html}
Uso: python 06_grupos_perenes.py <lista> [K]
"""
import json
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape
from sklearn.cluster import KMeans

from common import cfg, ver


c = cfg()
lista = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 12
CLS = [48, 41]
out = ver(c, "data", "rotulos", "grupos", "x").parent
out.mkdir(parents=True, exist_ok=True)
emb_root = ver(c, "data", "embeddings", "2025", "x").parent
regioes = [l.split() for l in ver(c, "data", "aoi", f"regions_{lista}.txt").read_text().split("\n") if l.strip()]


def mb_tile(src):  # mesma lógica do 05 (MapBiomas 30 m -> grade do tile, vizinho mais próximo)
    from rasterio.warp import Resampling, reproject, transform_bounds
    mb = np.zeros((src.height, src.width), np.uint8)
    with rasterio.open(c["externos"]["mapbiomas_2025"]) as m:
        b = transform_bounds(src.crs, m.crs, *src.bounds, densify_pts=21)
        w = m.window(*b).round_offsets().round_lengths()
        arr = m.read(1, window=w, boundless=True, fill_value=0)
        reproject(arr, mb, src_transform=m.window_transform(w), src_crs=m.crs,
                  dst_transform=src.transform, dst_crs=src.crs, resampling=Resampling.nearest)
    return mb


feats = []
for rid, crs in regioes:
    for t in sorted((emb_root / rid).glob("*.tif")):
        with rasterio.open(t) as s:
            a = s.read()
            valid = a[0] != 0
            if not valid.any():
                continue
            mb = mb_tile(s)
            m = ndimage.binary_opening(np.isin(mb, CLS) & valid, iterations=1)
            if m.sum() < 200:
                continue
            lab, n = ndimage.label(m)
            x = (a.astype(np.float32) - 128) / 127
            for geom, v in shapes(lab.astype(np.int32), mask=m, transform=s.transform):
                g = shape(geom)
                if g.area < 2e4:
                    continue
                px = lab == int(v)
                e = x[:, px].mean(1)
                feats.append({"regiao": rid, "crs": crs, "area_ha": g.area / 1e4,
                              "mb": int(np.bincount(mb[px]).argmax()), "emb": e / np.linalg.norm(e),
                              "geometry": gpd.GeoSeries([g], crs=crs).to_crs(4326).iloc[0]})
    print(f"{rid}: {sum(f['regiao'] == rid for f in feats)} manchas", flush=True)

g = gpd.GeoDataFrame(feats, crs=4326)
E = np.stack(g["emb"].values)
km = KMeans(K, n_init=10, random_state=42).fit(E, sample_weight=np.sqrt(g["area_ha"].values))
g["grupo"] = km.labels_
g["dist"] = 1 - (E * km.cluster_centers_[km.labels_]).sum(1)
np.save(out / f"{lista}_centroides.npy", km.cluster_centers_)
g.drop(columns="emb").to_file(out / f"{lista}_manchas.gpkg", driver="GPKG")

reps = []
for k, gr in g.groupby("grupo"):
    cand = gr[gr["area_ha"] >= 3].sort_values("dist")
    cand = cand.drop_duplicates("regiao").head(2) if len(cand.drop_duplicates("regiao")) >= 2 else cand.head(2)
    for j, (_, r) in enumerate(cand.iterrows()):
        p = r.geometry.representative_point()
        reps.append({"id": f"g{k:02d}{'ab'[j]}", "grupo": k, "regiao": r["regiao"], "lon": p.x, "lat": p.y,
                     "area_ha": round(r["area_ha"], 1), "mb": r["mb"],
                     "grupo_n": len(gr), "grupo_ha": round(gr["area_ha"].sum()),
                     "geojson": json.dumps(r.geometry.__geo_interface__)})
reps = pd.DataFrame(reps)
reps.drop(columns="geojson").to_csv(out / f"{lista}_reps.csv", index=False)

# visualizador: 12 mini-mapas por página (Esri World Imagery z18)
TPL = """<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>body{{margin:0;font:12px sans-serif;background:#111;color:#fff}}#g{{display:grid;grid-template-columns:repeat(4,300px);gap:4px}}
.c{{position:relative}}.m{{width:300px;height:300px}}.l{{position:absolute;top:2px;left:2px;z-index:999;background:#000a;padding:1px 4px}}</style>
</head><body><div id="g"></div><script>
const R={data};const U='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}';
R.forEach((r,i)=>{{const d=document.createElement('div');d.className='c';d.innerHTML=`<div class=l>${{r.id}} ${{r.regiao.split('_')[1]}} ${{r.area_ha}}ha mb${{r.mb}}</div><div class=m id=m${{i}}></div>`;
g.appendChild(d);const m=L.map('m'+i,{{zoomControl:false,attributionControl:false}}).setView([r.lat,r.lon],18);
L.tileLayer(U,{{maxZoom:19,maxNativeZoom:17}}).addTo(m);L.geoJSON(JSON.parse(r.geojson),{{style:{{color:'#ff0',weight:1,fill:false}}}}).addTo(m);}});
</script></body></html>"""
for i in range(0, len(reps), 12):
    pg = reps.iloc[i:i + 12][["id", "regiao", "lon", "lat", "area_ha", "mb", "geojson"]]
    (out / f"{lista}_visual_{i // 12 + 1}.html").write_text(TPL.format(data=pg.to_json(orient="records")), encoding="utf-8")
print(f"{len(g)} manchas, {g['area_ha'].sum():,.0f} ha, K={K}; {len(reps)} representantes; "
      f"{(len(reps) + 11) // 12} páginas em {out}")
print(g.groupby("grupo").agg(n=("area_ha", "size"), ha=("area_ha", "sum")).round(0).T.to_string())
