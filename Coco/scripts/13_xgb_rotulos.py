"""13 — XGBoost regional (GPU) + rótulos automáticos por região (receita caju v10/v10b adaptada).

Macro-regiões: 'sf' (lista sao_francisco) e 'litoral' (demais). Atributos: 64 embeddings + média 3×3 (128).
Positivos (treino XGB): r_ouro coco (ouro+prata) + candidatos confiáveis pelo QC (litoral: alta/média/limiar;
SF: média) erodidos 20 m + Descals (só litoral), sempre fora das classes MapBiomas seguras.
Negativos: r_ouro nao_coco + controles negativos + MapBiomas seguras + caju v10 ≥ 80.
Rótulo final {0 não-coco, 1 coco, 255 ignorado}:
  coco = XGB ≥ 0,70, fora de MB segura e de caju ≥ 50, manchas ≥ 1 ha, teto 1,2 × max(PAM, Censo) no município;
  não  = MB segura | caju ≥ 80 | (XGB ≤ 0,10 e Descals = 0);  r_ouro sobrepõe tudo; anel de 2 px em volta do coco = 255.
Saídas: v1/data/fontes/<r>_xgb.tif, v1/data/rotulos/auto/<r>_rotulo.tif, v1/reports/rotulos_auto.md
Uso: python 13_xgb_rotulos.py <lista>
"""
import sys

import cupy as cp
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xgboost as xgb
from rasterio.features import rasterize
from scipy import ndimage

from common import cfg, ext, ver

c = cfg()
R = c["rotulos"]
lista = sys.argv[1]
SEG = R["classes_seguras_nao_coco_mb"]
COCO_MB = R.get("classes_coco_mb", [48, 21, 41])
sf = {str(x) for x in c["listas_municipios"]["sao_francisco"]}
emb = ver(c, "data", "embeddings", "2025", "x").parent
fon = ver(c, "data", "fontes", "x").parent
rot = ver(c, "data", "rotulos", "x").parent
(rot / "auto").mkdir(exist_ok=True)
regioes = [l.split() for l in ver(c, "data", "aoi", f"regions_{lista}.txt").read_text().split("\n") if l.strip()]
macro = lambda rid: "sf" if rid.split("_")[0] in sf else "litoral"
ouro = gpd.read_file(rot / "r_ouro.gpkg")
cand = {"litoral": gpd.read_file(rot / "candidatos" / "candidatos_top15_caatinga.gpkg"),
        "sf": gpd.read_file(rot / "candidatos" / "candidatos_sao_francisco_modelo_sao_francisco_g11.gpkg")}
conf = {"litoral": ["alta", "media", "limiar"], "sf": ["media"]}
rng = np.random.default_rng(42)


def feats(a):
    x = (a.astype(np.float32) - 128) / 127
    m = ndimage.uniform_filter(x, size=(1, 3, 3))
    return np.concatenate([x, m]).reshape(128, -1).T, a[0] != 0


def fonte(rid, k, win):
    with rasterio.open(fon / f"{rid}_{k}.tif") as s:
        return s.read(1, window=win)


def tiles(rid):
    with rasterio.open(fon / f"{rid}_mb.tif") as g:
        gt = g.transform
    for t in sorted((emb / rid).glob("*.tif")):
        with rasterio.open(t) as s:
            col, row = round((s.bounds.left - gt.c) / 10), round((gt.f - s.bounds.top) / 10)
            yield s, rasterio.windows.Window(col, row, s.width, s.height)


def vetores(rid, crs, tr, shp):
    m = macro(rid)
    o = ouro[ouro.nivel == "ouro"].to_crs(crs)   # v1b: prata (grupos inteiros, heterogêneos) fora — QC 21/11
    cd = cand[m].to_crs(crs)
    pos = list(o[o.classe == "coco"].geometry) + list(cd[cd.faixa.isin(conf[m])].geometry.buffer(-20))
    neg = list(o[o.classe == "nao_coco"].geometry) + list(cd[cd.faixa == "controle_neg"].geometry)
    rz = lambda gs: rasterize([(g, 1) for g in gs if not g.is_empty], out_shape=shp, transform=tr, fill=0).astype(bool) \
        if any(not g.is_empty for g in gs) else np.zeros(shp, bool)
    return rz(pos), rz(neg), rz(list(o[o.classe == "coco"].geometry)), rz(list(o[o.classe == "nao_coco"].geometry))


# ---------- 1. amostras de treino ----------
# Otimização (21/11/2026): fontes e vetores rasterizados UMA vez por região (antes: a cada tile) e
# amostragem em todos os tiles com positivos + fração aleatória dos demais (FRAC_NEG). Ver DECISOES.md.
FRAC_NEG = R.get("frac_tiles_negativos", 0.4)
X, Y = {"litoral": [], "sf": []}, {"litoral": [], "sf": []}
MF = {m: ver(c, "models", f"xgb_rotulos_{m}.json") for m in X}
REUSA = all(f.exists() for f in MF.values()) and "--refazer" not in sys.argv  # retoma: pula amostragem
for rid, crs in ([] if REUSA else regioes):
    m = macro(rid)
    with rasterio.open(fon / f"{rid}_mb.tif") as g:
        gtr, gshp = g.transform, g.shape
        MB = g.read(1)
    DS, CJ = (rasterio.open(fon / f"{rid}_{k}.tif").read(1) for k in ["descals", "caju"])
    POS, NEG, _, _ = vetores(rid, crs, gtr, gshp)
    for s, w in tiles(rid):
        sl = (slice(w.row_off, w.row_off + w.height), slice(w.col_off, w.col_off + w.width))
        pos = POS[sl].ravel()
        if not pos.any() and not (m == "litoral" and DS[sl].any()) and rng.random() > FRAC_NEG:
            continue
        F, valid = feats(s.read())
        mb, ds, cj = MB[sl].ravel(), DS[sl].ravel(), CJ[sl].ravel()
        neg = NEG[sl].ravel()
        seg = np.isin(mb, SEG)
        p = valid.ravel() & np.isin(mb, COCO_MB) & (pos | ((ds == 1) if m == "litoral" else False))
        n = valid.ravel() & (neg | seg | ((cj >= 80) & (cj != 255)))
        for msk, y, k in [(p, 1, 600), (n, 0, 400)]:
            idx = np.flatnonzero(msk)
            if idx.size:
                idx = rng.choice(idx, min(k, idx.size), replace=False)
                X[m].append(F[idx]); Y[m].append(np.full(idx.size, y, np.uint8))
    print(f"amostras {rid}: pos={sum((y == 1).sum() for y in Y[m])} neg={sum((y == 0).sum() for y in Y[m])} ({m})", flush=True)

bst = {m: xgb.Booster(model_file=str(f)) for m, f in MF.items()} if REUSA else {}
for _b in bst.values():
    _b.set_param({"device": "cuda"})  # senão a predição cai na CPU (lição do caju)
for m in X:
    if not X[m]:
        continue
    Xa, Ya = np.concatenate(X[m]), np.concatenate(Y[m])
    dm = xgb.DMatrix(cp.asarray(Xa), label=cp.asarray(Ya))
    bst[m] = xgb.train({"objective": "binary:logistic", "tree_method": "hist", "device": "cuda", "max_depth": 8,
                        "eta": 0.1, "subsample": 0.8, "colsample_bytree": 0.8,
                        "scale_pos_weight": float((Ya == 0).sum() / max(1, (Ya == 1).sum()))}, dm, 400)
    bst[m].save_model(str(ver(c, "models", f"xgb_rotulos_{m}.json")))
    print(f"XGB {m}: {len(Ya)} amostras ({int(Ya.sum())} pos)", flush=True)

# ---------- 2. predição + rótulos ----------
pc = pd.read_csv(ext(c, "pam", "pam_censo_coco.csv"), dtype={"cod": str})
mun = gpd.read_file(c["externos"]["municipios_shp"])
colc = next(k for k in mun.columns if k.upper().startswith("CD_MUN"))
linhas = []
for rid, crs in regioes:
    with rasterio.open(fon / f"{rid}_mb.tif") as g:
        prof, tr, shp = g.profile, g.transform, g.shape
        mb = g.read(1)
    ds, cj = (rasterio.open(fon / f"{rid}_{k}.tif").read(1) for k in ["descals", "caju"])
    pr = np.zeros(shp, np.uint8)
    valid = np.zeros(shp, bool)
    for s, w in tiles(rid):
        F, v = feats(s.read())
        p = bst[macro(rid)].inplace_predict(cp.asarray(F)).get()
        sl = (slice(w.row_off, w.row_off + w.height), slice(w.col_off, w.col_off + w.width))
        pr[sl] = np.where(v.ravel(), np.round(p * 100), 0).reshape(s.height, s.width)
        valid[sl] = v.reshape(s.height, s.width)
    with rasterio.open(fon / f"{rid}_xgb.tif", "w", **prof) as o:
        o.write(pr, 1)
    seg = np.isin(mb, SEG)
    cjv = (cj != 255)
    coco = (pr >= 70) & ~seg & ~(cjv & (cj >= 50)) & valid
    lab, n = ndimage.label(coco)
    tam = np.bincount(lab.ravel())
    coco &= (tam >= R["mancha_min_ha"] * 100)[lab]
    # teto por município
    mpoly = mun[mun[colc].astype(str) == rid.split("_")[0]].to_crs(crs)
    mm = rasterize([(g, 1) for g in mpoly.geometry], out_shape=shp, transform=tr, fill=0).astype(bool)
    row = pc[pc.cod == rid.split("_")[0]]
    ref = float(np.nanmax([row.area_destinada_media.iloc[0], row.area_existente_ha.iloc[0]])) if len(row) else np.nan
    teto = R["teto_pam"] * ref * 100  # em pixels
    cm = coco & mm
    if np.isfinite(teto) and cm.sum() > teto:
        corte = np.sort(pr[cm])[::-1][int(teto)]
        coco &= pr > corte
    nao = valid & (seg | (cjv & (cj >= 80)) | ((pr <= 10) & (ds == 0)))
    rot_ = np.full(shp, 255, np.uint8)
    rot_[nao] = 0
    rot_[coco] = 1
    anel = ndimage.binary_dilation(coco, iterations=R["anel_ignorado_px"]) & ~coco
    rot_[anel & (rot_ == 0)] = 255
    _, _, oc, on = vetores(rid, crs, tr, shp)
    rot_[on] = 0
    rot_[oc] = 1
    rot_[~valid] = 255
    with rasterio.open(rot / "auto" / f"{rid}_rotulo.tif", "w", **prof) as o:
        o.write(rot_, 1)
    ha = lambda a: a.sum() / 100
    linhas.append(f"| {rid} | {ha((rot_ == 1) & mm):,.0f} | {ref:,.0f} | {ha((rot_ == 1) & mm) / ref if ref else np.nan:.2f} | "
                  f"{ha((rot_ == 0) & mm):,.0f} | {100 * ((rot_ == 255) & mm).sum() / max(1, mm.sum()):.0f}% | {ha(oc):,.0f} |")
    print(linhas[-1], flush=True)

ver(c, "reports", "rotulos_auto.md").write_text("\n".join([
    "# Rótulos automáticos v1", "", "Coco (ha) no município × referência = max(PAM área destinada 2023–25, Censo 2017 área existente).", "",
    "| Região | Coco rotulado (ha) | Referência (ha) | Razão | Não-coco (ha) | Ignorado | R-ouro/prata coco (ha) |",
    "|---|---|---|---|---|---|---|", *linhas]), encoding="utf-8")
