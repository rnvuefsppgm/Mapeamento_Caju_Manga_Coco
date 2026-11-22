"""20 — Chips 256 px + cache memmap (formato do caju v10b, já em uint8).

Seleção por região: todos os chips com R-ouro; até 120 com coco ≥ 1%; até 80 sem coco com ≥ 30% rotulado.
Grade sem sobreposição (aumento de dados no treino). Embeddings mantidos codificados (uint8: (v-128)/127; 0 = NoData).
Saídas em v1/data/cache/: features.npy (N,64,256,256) uint8, labels.npy (N,256,256) rótulo de treino,
labels_ouro.npy (N,256,256) só R-ouro verificado (avaliação; 255 = sem rótulo), chips_metadata.parquet
(chip_id, regiao, macro, block_id 10 km, source=field|pseudo, coco_fraction, labeled_fraction, ouro_px, cache_idx).
Uso: python 20_chips_cache.py <lista>
"""
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.merge import merge

from common import cfg, ver

c = cfg()
lista = sys.argv[1]
S = c["chips"]["size"]
BK = c["treino"]["bloco_km"] * 1000
sf = {str(x) for x in c["listas_municipios"]["sao_francisco"]}
emb = ver(c, "data", "embeddings", "2025", "x").parent
rot = ver(c, "data", "rotulos", "x").parent
out = ver(c, "data", "cache", "x").parent
out.mkdir(parents=True, exist_ok=True)
regioes = [l.split() for l in ver(c, "data", "aoi", f"regions_{lista}.txt").read_text().split("\n") if l.strip()]
ouro = gpd.read_file(rot / "r_ouro.gpkg")
ouro = ouro[ouro.nivel == "ouro"]
rng = np.random.default_rng(42)

sel = []  # (rid, crs, r0, c0, flags)
for rid, crs in regioes:
    with rasterio.open(rot / "auto" / f"{rid}_rotulo.tif") as s:
        L, tr, H, W = s.read(1), s.transform, s.height, s.width
    o = ouro.to_crs(crs)
    O = np.full((H, W), 255, np.uint8)
    for cl, v in [("nao_coco", 0), ("coco", 1)]:
        g = o[o.classe == cl].geometry
        if len(g):
            O[rasterize([(x, 1) for x in g], out_shape=(H, W), transform=tr, fill=0).astype(bool)] = v
    oy, ox = rng.integers(0, S), rng.integers(0, S)
    cand = []
    for r0 in range(oy, H - S + 1, S):
        for c0 in range(ox, W - S + 1, S):
            l = L[r0:r0 + S, c0:c0 + S]
            cand.append((r0, c0, (l == 1).mean(), (l != 255).mean(), int((O[r0:r0 + S, c0:c0 + S] != 255).sum())))
    d = pd.DataFrame(cand, columns=["r0", "c0", "coco_fraction", "labeled_fraction", "ouro_px"])
    f = d[d.ouro_px > 0]
    p = d[(d.ouro_px == 0) & (d.coco_fraction >= 0.01)]
    n = d[(d.ouro_px == 0) & (d.coco_fraction == 0) & (d.labeled_fraction >= c["chips"]["min_rotulado"])]
    pick = pd.concat([f.assign(source="field"), p.sample(min(120, len(p)), random_state=1).assign(source="pseudo"),
                      n.sample(min(80, len(n)), random_state=1).assign(source="pseudo")])
    for _, r in pick.iterrows():
        sel.append(dict(r, regiao=rid, crs=crs, macro="sf" if rid.split("_")[0] in sf else "litoral"))
    print(f"{rid}: {len(f)} campo + {min(120, len(p))} coco + {min(80, len(n))} não-coco", flush=True)

meta = pd.DataFrame(sel).reset_index(drop=True)
N = len(meta)
F = np.lib.format.open_memmap(out / "features.npy", mode="w+", dtype="uint8", shape=(N, 64, S, S))
LB = np.lib.format.open_memmap(out / "labels.npy", mode="w+", dtype="uint8", shape=(N, S, S))
LO = np.lib.format.open_memmap(out / "labels_ouro.npy", mode="w+", dtype="uint8", shape=(N, S, S))
bids = []
for rid, grp in meta.groupby("regiao", sort=False):
    crs = grp.crs.iloc[0]
    with rasterio.open(rot / "auto" / f"{rid}_rotulo.tif") as s:
        L, tr = s.read(1), s.transform
    o = ouro.to_crs(crs)
    O = np.full(L.shape, 255, np.uint8)
    for cl, v in [("nao_coco", 0), ("coco", 1)]:
        g = o[o.classe == cl].geometry
        if len(g):
            O[rasterize([(x, 1) for x in g], out_shape=L.shape, transform=tr, fill=0).astype(bool)] = v
    tiles = [rasterio.open(t) for t in sorted((emb / rid).glob("*.tif"))]
    for i, r in grp.iterrows():
        x0, y0 = tr * (r.c0, r.r0)
        a, _ = merge(tiles, bounds=(x0, y0 - S * 10, x0 + S * 10, y0), res=10, nodata=0)
        a = a[:, :S, :S]
        F[i] = a
        lab = L[r.r0:r.r0 + S, r.c0:r.c0 + S].copy()
        lab[a[0] == 0] = 255
        LB[i] = lab
        LO[i] = O[r.r0:r.r0 + S, r.c0:r.c0 + S]
        cx, cy = x0 + S * 5, y0 - S * 5
        meta.loc[i, "block_id"] = f"{crs[-2:]}_{int(cx // BK)}_{int(cy // BK)}"
    for t in tiles:
        t.close()
F.flush(); LB.flush(); LO.flush()
meta["cache_idx"] = np.arange(N)
meta["chip_id"] = [f"{r.regiao}_{r.r0}_{r.c0}" for r in meta.itertuples()]
meta.drop(columns=["crs"]).to_parquet(out / "chips_metadata.parquet", index=False)
print(f"Cache: {N} chips ({(meta.source == 'field').sum()} campo), {N * 64 * S * S / 1e9:.1f} GB em {out}")
print(meta.groupby("macro").agg(chips=("chip_id", "size"), blocos=("block_id", "nunique"),
                                coco=("coco_fraction", "mean"), campo=("source", lambda s: (s == "field").sum())).round(3).to_string())
