"""01 (v10) — Fontes independentes por município, na GPU (só embeddings 2025, 10 m).

S1 U-Net   : probabilidade do mapa v8 (já existe).
S2 XGBoost : árvore por pixel (64 embeddings + média 3x3 = 128 atributos), device=cuda. Treino em Serra do Mel:
             positivos = polígonos de campo (erodidos 1 px); negativos confiáveis = >100 m dos polígonos
             E (U-Net <= 0,10 OU classe segura MapBiomas). (aprendizado PU em 2 etapas)
S3 Similar.: maior cosseno com 8 "assinaturas" de caju (k-means GPU nos pixels dos polígonos); limiar T_sim
             = menor valor com precisão >= 0,90 na amostra de Serra do Mel.
S4 Grupos  : k-means GPU com 40 grupos POR REGIÃO (litoral CE / sertão CE-RN / PI-PE) nos embeddings.
Saídas: data/fontes/<regiao>_{s2,s3,cl}.tif (uint8), reports/cluster_stats.csv, reports/fontes_resumo.json.
MapBiomas (API, v9/data/mapbiomas_api) é usada se existir; senão segue sem ela.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
import xgboost as xgb
import cupy as cp  # predição XGBoost direto na GPU (cupy-cuda13x[ctk])
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from scipy.ndimage import binary_erosion

ROOT = Path(__file__).parent
V8, V9 = ROOT.parent / "v8", ROOT.parent / "v9"
EMB, PRED, OUT = V8 / "data/ee_exports/2025/embeddings", V8 / "predictions/2025", ROOT / "data/fontes"
DEV = "cuda"
NEG_MB, PERENE = [3, 5, 23, 24, 25, 30, 31, 33], 36
REG = {"litoral_CE": [2302305, 2302206, 2303501, 2301109, 2304251, 2304459, 2307809, 2305357, 2306405,
                      2306553, 2307254, 2304707],
       "sertao_CE_RN": [2300705, 2309458, 2301950, 2303956, 2309607, 2301208, 2401008, 2402709, 2413359, 2413607],
       "PI_PE": [2208205, 2206506, 2602803]}
REG_OF = {c: r for r, cs in REG.items() for c in cs}
BLK = 1024
torch.manual_seed(0); rng = np.random.default_rng(0)


def rids():
    return [r for r in (V8 / "data/aoi/regions.txt").read_text().split() if r.startswith("m")]


def mb_raster(rid, shape, tr, crs):
    f = sorted((V9 / "data/mapbiomas_api").glob(f"{rid}_*.tif"))
    out = np.zeros(shape, "uint8")
    if f:
        with rasterio.open(f[-1]) as m:
            reproject(rasterio.band(m, 1), out, dst_transform=tr, dst_crs=crs, dst_nodata=0,
                      resampling=Resampling.nearest)
    return out


def polys(shape, tr, crs, buf=0.0):
    g = gpd.read_file(ROOT.parent / "v7/data/vectors/Pol_Caju.shp").to_crs(crs)
    geoms = [x for x in (g.buffer(buf) if buf else g).geometry if x is not None and not x.is_empty]
    return rasterize([(x, 1) for x in geoms], out_shape=shape, transform=tr, fill=0).astype(bool) if geoms else np.zeros(shape, bool)


def blocks(H, W):
    for r0 in range(0, H, BLK):
        for c0 in range(0, W, BLK):
            yield r0, c0, min(BLK, H - r0), min(BLK, W - c0)


def feats(src, r0, c0, h, w):
    """(h*w,128) na GPU + máscara válida; média 3x3 com margem de 1 px."""
    rr, cc = max(r0 - 1, 0), max(c0 - 1, 0)
    win = Window(cc, rr, min(c0 + w + 1, src.width) - cc, min(r0 + h + 1, src.height) - rr)
    u = torch.from_numpy(src.read(window=win)).to(DEV)
    x = (u.float() - 128.0) / 127.0
    mu = F.avg_pool2d(F.pad(x[None], (1, 1, 1, 1), mode="replicate"), 3, 1)[0]
    ir, ic = r0 - rr, c0 - cc
    x, mu, u = x[:, ir:ir + h, ic:ic + w], mu[:, ir:ir + h, ic:ic + w], u[:, ir:ir + h, ic:ic + w]
    return torch.cat([x, mu]).reshape(128, -1).T.contiguous(), (u != 0).all(0).reshape(-1)


def kmeans(X, k, it=25):
    C = X[torch.randperm(len(X), device=X.device)[:k]].clone()
    for _ in range(it):
        a = torch.cdist(X, C).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                C[j] = X[m].mean(0)
    return C


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CK = ROOT / "models" / "fontes_treino.pt"
    if CK.exists():  # retomada: reaproveita treino S2/S3/S4
        ck = torch.load(CK, weights_only=False)
        bst = xgb.Booster(); bst.load_model(str(ROOT / "models" / "s2_xgb.json")); bst.set_param({"device": "cuda"})
        proto, cent, tsim, rep = ck["proto"].to(DEV), {k: v.to(DEV) for k, v in ck["cent"].items()}, ck["tsim"], ck["rep"]
        return aplicar(bst, proto, cent, tsim, rep)
    rep = {}
    # ---------- treino S2/S3 em Serra do Mel ----------
    sm = "m2413359_SerradoMel"
    with rasterio.open(EMB / f"{sm}.vrt") as src, rasterio.open(PRED / f"{sm}_prob.tif") as pu:
        H, W, tr, crs = src.height, src.width, src.transform, src.crs
        U = pu.read(1)
        pos = binary_erosion(polys((H, W), tr, crs))
        near = polys((H, W), tr, crs, buf=100.0)
        MB = mb_raster(sm, (H, W), tr, crs)
        neg = ~near & (((U <= 10) & (U != 255)) | np.isin(MB, NEG_MB))
        Xp, Xn = [], []
        for r0, c0, h, w in blocks(H, W):
            X, v = feats(src, r0, c0, h, w)
            p = torch.from_numpy(pos[r0:r0 + h, c0:c0 + w].reshape(-1)).to(DEV) & v
            n = torch.from_numpy(neg[r0:r0 + h, c0:c0 + w].reshape(-1)).to(DEV) & v
            for msk, L, q in ((p, Xp, 0.15), (n, Xn, 0.02)):
                idx = torch.nonzero(msk).squeeze(1)
                if len(idx):
                    L.append(X[idx[torch.rand(len(idx), device=DEV) < q]].cpu())
    Xp, Xn = torch.cat(Xp)[:300000], torch.cat(Xn)[:300000]
    rep["amostras_treino"] = {"pos": len(Xp), "neg": len(Xn)}
    Xtr = torch.cat([Xp, Xn]).numpy(); ytr = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
    clf = xgb.XGBClassifier(device="cuda", tree_method="hist", n_estimators=400, max_depth=8,
                            learning_rate=0.05, subsample=0.8, colsample_bytree=0.5)
    clf.fit(Xtr, ytr)
    bst = clf.get_booster()
    bst.set_param({"device": "cuda"})  # predição na GPU (sem isso caía na CPU, muito lento)
    # S3: assinaturas (só os 64 embeddings, normalizados)
    P64 = F.normalize(Xp[:, :64].to(DEV), dim=1)
    proto = F.normalize(kmeans(P64, 8), dim=1)
    simp = (P64 @ proto.T).max(1).values.cpu().numpy()
    simn = (F.normalize(Xn[:, :64].to(DEV), dim=1) @ proto.T).max(1).values.cpu().numpy()
    tsim = 0.99
    for t in np.arange(0.5, 1.0, 0.005):
        tp, fp = (simp >= t).sum(), (simn >= t).sum()
        if tp + fp and tp / (tp + fp) >= 0.90:
            tsim = float(t); break
    rep["T_sim"] = round(tsim, 3)
    rep["S3_recall_pos_SM"] = round(float((simp >= tsim).mean()), 3)
    # ---------- S4: k-means por região ----------
    cent = {}
    for reg, cods in REG.items():
        smp = []
        for rid in [r for r in rids() if int(r[1:8]) in cods]:
            with rasterio.open(EMB / f"{rid}.vrt") as src:
                for r0, c0, h, w in blocks(src.height, src.width):
                    if rng.random() > 0.35:
                        continue
                    X, v = feats(src, r0, c0, h, w)
                    X = X[v][:, :64]
                    if len(X):
                        smp.append(F.normalize(X[torch.randperm(len(X), device=DEV)[:4000]], dim=1).cpu())
        cent[reg] = kmeans(torch.cat(smp).to(DEV), 40)
    bst.save_model(str(ROOT / "models" / "s2_xgb.json"))
    torch.save({"proto": proto.cpu(), "cent": {k: v.cpu() for k, v in cent.items()}, "tsim": tsim, "rep": rep}, CK)
    return aplicar(bst, proto, cent, tsim, rep)


def aplicar(bst, proto, cent, tsim, rep):
    """S2/S3/S4 em cada município + estatística dos grupos (retomável por município)."""
    sd = ROOT / "reports" / "cluster_stats"; sd.mkdir(parents=True, exist_ok=True)
    for rid in rids():
        if (sd / f"{rid}.csv").exists():
            continue
        cod = int(rid[1:8]); reg = REG_OF[cod]
        with rasterio.open(EMB / f"{rid}.vrt") as src, rasterio.open(PRED / f"{rid}_prob.tif") as pu:
            H, W, tr, crs = src.height, src.width, src.transform, src.crs
            U = pu.read(1); MB = mb_raster(rid, (H, W), tr, crs); PL = polys((H, W), tr, crs)
            S2 = np.full((H, W), 255, "uint8"); S3 = S2.copy(); CL = S2.copy()
            for r0, c0, h, w in blocks(H, W):
                X, v = feats(src, r0, c0, h, w)
                s2 = torch.from_dlpack(bst.inplace_predict(cp.from_dlpack(X.contiguous()))).float()
                e = F.normalize(X[:, :64], dim=1)
                s3 = (e @ proto.T).max(1).values
                cl = torch.cdist(e, cent[reg]).argmin(1)
                for A, t in ((S2, s2.clamp(0, 1) * 100), (S3, s3.clamp(0, 1) * 100), (CL, cl.float())):
                    t = torch.where(v, t.round(), torch.full_like(t, 255))
                    A[r0:r0 + h, c0:c0 + w] = t.reshape(h, w).byte().cpu().numpy()
            ok = (CL != 255) & (U != 255)
            df = pd.DataFrame({"cl": CL[ok], "u": U[ok] / 100, "s2": S2[ok] / 100, "s3ok": S3[ok] >= tsim * 100,
                               "per": MB[ok] == PERENE, "neg": np.isin(MB[ok], NEG_MB), "pol": PL[ok]})
            g = df.groupby("cl").agg(n=("u", "size"), u=("u", "sum"), s2=("s2", "sum"), s3=("s3ok", "sum"),
                                     per=("per", "sum"), neg=("neg", "sum"), pol=("pol", "sum")).reset_index()
            g["regiao"], g["municipio"] = reg, rid
            g.to_csv(sd / f"{rid}.csv", index=False)
            prof = dict(driver="GTiff", height=H, width=W, count=1, dtype="uint8", crs=crs, transform=tr,
                        nodata=255, tiled=True, blockxsize=512, blockysize=512, compress="deflate")
            for A, s in ((S2, "s2"), (S3, "s3"), (CL, "cl")):
                with rasterio.open(OUT / f"{rid}_{s}.tif", "w", **prof) as d:
                    d.write(A, 1)
            print(rid, "ok", flush=True)
    pd.concat([pd.read_csv(x) for x in sorted(sd.glob("m*.csv"))]).to_csv(ROOT / "reports/cluster_stats.csv", index=False)
    rep["mapbiomas_disponivel"] = len(list((V9 / "data/mapbiomas_api").glob("*.tif")))
    (ROOT / "reports/fontes_resumo.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(rep)


if __name__ == "__main__":
    main()
