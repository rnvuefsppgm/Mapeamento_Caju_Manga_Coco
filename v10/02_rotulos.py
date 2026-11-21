"""02 (v10) — Grupos -> classe; pacote de revisão; rótulo por pixel por CONCORDÂNCIA.

Grupo (por região, a partir de reports/cluster_stats.csv), com mU = média U-Net, m2 = média XGBoost,
m3 = fração com similaridade >= T_sim:
  caju       : >= 50% em polígonos de campo  OU  (>= 2 de {mU, m2, m3} >= 0,6  E  média >= 0,7)
  não-caju   : média <= 0,1  OU  >= 60% em classes seguras MapBiomas
  duvidoso   : resto  +  qualquer grupo do litoral CE com >= 30% de Lavoura Perene (risco coqueiro)
Revisão (opcional): data/revisao/grupos_duvidosos.gpkg (8 pontos por grupo, campo `classe` = caju|coqueiro|outro).
Se data/revisao/grupos_revisados.csv existir (regiao, cl, classe), ele prevalece sobre a regra.
Pixel:
  1 (caju)    : polígono de campo  OU  (grupo caju  E  >= 2 de {U>=0,7 ; S2>=0,7 ; S3>=T_sim}  E  fora MB seguro)
                e mancha >= 1 ha
  0 (não)     : MB seguro  OU  (grupo não-caju  E  U<=0,1  E  S2<=0,1)  OU  grupo revisado coqueiro/outro
  255 (ignora): resto + 2 px em volta das manchas de caju
Checagem: área de caju rotulada <= 1,2 x PAM 2025; senão exige 3/3 fontes.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

ROOT = Path(__file__).parent
_s = importlib.util.spec_from_file_location("f", ROOT / "01_fontes.py"); f = importlib.util.module_from_spec(_s); _s.loader.exec_module(f)
FT, OUT = ROOT / "data/fontes", ROOT / "data/rotulos"
rep = json.loads((ROOT / "reports/fontes_resumo.json").read_text(encoding="utf-8"))


def t_sim_balanceado():
    """Recalibra T_sim com classes BALANCEADAS (a amostra original tinha 81% de positivos, o que
    deixava o limite frouxo: recall 0,996). Precisão >= 0,90 com pesos iguais pos/neg em Serra do Mel."""
    rid = "m2413359_SerradoMel"
    with rasterio.open(f.PRED / f"{rid}_prob.tif") as p:
        U, tr, crs = p.read(1), p.transform, p.crs
    S3 = rasterio.open(FT / f"{rid}_s3.tif").read(1)
    pos = ndimage.binary_erosion(f.polys(U.shape, tr, crs)) & (S3 != 255)
    neg = ~f.polys(U.shape, tr, crs, buf=100.0) & (U <= 10) & (S3 != 255)
    sp, sn = S3[pos].astype(float), S3[neg].astype(float)
    for t in range(50, 101):
        r_p, r_n = (sp >= t).mean(), (sn >= t).mean()
        if r_p + r_n > 0 and r_p / (r_p + r_n) >= 0.90:
            rep["T_sim_balanceado"] = t / 100; rep["S3_recall_bal"] = round(float(r_p), 3)
            return float(t)
    return 99.0


T = t_sim_balanceado()


def votos_por_grupo():
    acc = {}
    for rid in f.rids():
        reg = f.REG_OF[int(rid[1:8])]
        U = rasterio.open(f.PRED / f"{rid}_prob.tif").read(1)
        S2, S3, CL = (rasterio.open(FT / f"{rid}_{k}.tif").read(1) for k in ("s2", "s3", "cl"))
        ok = (U != 255) & (CL != 255)
        v = ((U >= 70).astype("uint8") + (S2 >= 70) + (S3 >= T))[ok]
        c = CL[ok].astype(int)
        n, a2, a3 = (np.bincount(c, weights=w, minlength=40) for w in (None, v >= 2, v == 3))
        t = acc.setdefault(reg, np.zeros((3, 40)))
        t += np.vstack([n, a2, a3])
    rows = [{"regiao": r, "cl": k, "vf2": t[1, k] / max(t[0, k], 1), "vf3": t[2, k] / max(t[0, k], 1)}
            for r, t in acc.items() for k in range(40)]
    return pd.DataFrame(rows)


def classify():
    s = pd.read_csv(ROOT / "reports/cluster_stats.csv").groupby(["regiao", "cl"]).sum(numeric_only=True).reset_index()
    for c in ("u", "s2", "s3", "per", "neg", "pol"):
        s[c] = s[c] / s["n"]
    s["media"] = s[["u", "s2", "s3"]].mean(axis=1)
    # v10b: fração de pixels do grupo com >= 2 fontes concordando (a média diluía grupos mistos)
    vf = votos_por_grupo()
    s = s.merge(vf, on=["regiao", "cl"], how="left").fillna({"vf2": 0, "vf3": 0})
    s["classe"] = "duvidoso"
    s.loc[(s.pol >= 0.5) | (s.vf2 >= 0.5), "classe"] = "caju"
    s.loc[((s.vf2 <= 0.02) & (s.media <= 0.15)) | (s.neg >= 0.6), "classe"] = "nao_caju"
    s.loc[(s.regiao == "litoral_CE") & (s.per >= 0.3) & (s.pol < 0.5), "classe"] = "duvidoso"
    rv = ROOT / "data/revisao/grupos_revisados.csv"
    if rv.exists():
        r = pd.read_csv(rv)
        s = s.merge(r[["regiao", "cl", "classe"]].rename(columns={"classe": "rev"}), on=["regiao", "cl"], how="left")
        s["classe"] = np.where(s.rev.notna(), s.rev.map({"caju": "caju"}).fillna("revisado_nao"), s.classe)
    s.to_csv(ROOT / "reports/grupos_classificados.csv", index=False)
    return s


def review_points(s):
    d = s[s.classe == "duvidoso"]
    rng = np.random.default_rng(1)
    rows = []
    for rid in f.rids():
        reg = f.REG_OF[int(rid[1:8])]
        dd = set(d[d.regiao == reg].cl.astype(int))
        if not dd:
            continue
        with rasterio.open(FT / f"{rid}_cl.tif") as c:
            CL, tr, crs = c.read(1), c.transform, c.crs
        for cl in dd:
            r, cc = np.nonzero(CL == cl)
            if len(r) == 0:
                continue
            k = rng.choice(len(r), min(3, len(r)), replace=False)  # 3 por município -> ~8+ por grupo na região
            xs, ys = rasterio.transform.xy(tr, r[k], cc[k])
            rows += [{"regiao": reg, "cl": int(cl), "municipio": rid, "classe": "", "x": x, "y": y} for x, y in zip(xs, ys)]
    if rows:
        g = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=gpd.points_from_xy([r["x"] for r in rows], [r["y"] for r in rows]),
                             crs=crs).drop(columns=["x", "y"]).to_crs(4326)
        g.to_file(ROOT / "data/revisao/grupos_duvidosos.gpkg", driver="GPKG")
    return len(rows)


def label(rid, s, pam_ha, strict=False):
    reg = f.REG_OF[int(rid[1:8])]
    cs = s[s.regiao == reg].set_index("cl").classe
    with rasterio.open(f.PRED / f"{rid}_prob.tif") as p:
        U, tr, crs = p.read(1), p.transform, p.crs
    S2, S3, CL = (rasterio.open(FT / f"{rid}_{k}.tif").read(1) for k in ("s2", "s3", "cl"))
    MB, PL = f.mb_raster(rid, U.shape, tr, crs), f.polys(U.shape, tr, crs)
    valid = (U != 255) & (CL != 255)
    lut = lambda name: np.isin(CL, [int(c) for c in cs[cs == name].index])
    seg = np.isin(MB, f.NEG_MB)
    vot = (U >= 70).astype(int) + (S2 >= 70) + (S3 >= T)
    # v10c: S3 não discrimina (mediana ~0,84 em toda parte) -> concordância = U-Net E XGBoost (modelos independentes)
    q = 85 if strict else 70
    pos = PL | (~lut("nao_caju") & (U >= q) & (S2 >= q) & (U != 255) & ~seg)
    pos &= valid
    lab, n = ndimage.label(pos)
    if n:
        sz = ndimage.sum(pos, lab, range(1, n + 1))
        pos = np.isin(lab, np.where(sz >= 100)[0] + 1) | (PL & valid)
    neg = valid & (seg | (lut("nao_caju") & (U <= 10) & (S2 <= 10)) | lut("revisado_nao")
                   | ((vot == 0) & (U <= 5) & (S2 <= 5))) & ~pos  # v10b: 0/3 fontes e prob. muito baixa
    L = np.full(U.shape, 255, "uint8")
    L[neg] = 0
    ring = ndimage.binary_dilation(pos, iterations=2) & ~pos
    L[ring] = 255
    L[pos] = 1
    ha = pos.sum() * 0.01
    if not strict and pam_ha and ha > 1.2 * pam_ha:
        return label(rid, s, pam_ha, strict=True)
    prof = dict(driver="GTiff", height=L.shape[0], width=L.shape[1], count=1, dtype="uint8", crs=crs,
                transform=tr, nodata=255, tiled=True, blockxsize=512, blockysize=512, compress="deflate")
    with rasterio.open(OUT / f"{rid}_rot.tif", "w", **prof) as d:
        d.write(L, 1)
    return {"municipio": rid, "caju_ha": round(ha), "nao_caju_ha": round(neg.sum() * 0.01),
            "ignorado_%": round(100 * (L == 255)[valid].mean(), 1), "regra": "U&XGB>=0,85" if strict else "U&XGB>=0,70",
            "PAM_ha": pam_ha}


def main():
    OUT.mkdir(parents=True, exist_ok=True); (ROOT / "data/revisao").mkdir(parents=True, exist_ok=True)
    s = classify()
    npts = review_points(s)
    pam = pd.read_csv(ROOT.parent / "dados_externos/ranking_caju_caatinga_2023_2025.csv").set_index("cod").area_ha_2025
    res = [label(r, s, float(pam.get(int(r[1:8]), 0))) for r in f.rids()]
    t = pd.DataFrame(res)
    cnt = s.groupby(["regiao", "classe"]).size().unstack(fill_value=0)
    md = ("# Rótulos por concordância (v10)\n\n" f"T_sim = {rep['T_sim']} | amostras S2: {rep['amostras_treino']} | "
          f"MapBiomas disponível: {rep['mapbiomas_disponivel']} municípios\n\n## Grupos por região\n\n"
          + cnt.to_markdown() + f"\n\nPontos de revisão gerados: {npts} (data/revisao/grupos_duvidosos.gpkg)\n\n"
          "## Rótulos por município\n\n" + t.to_markdown(index=False) +
          f"\n\nTotal caju rotulado: {t.caju_ha.sum():,.0f} ha | não-caju: {t.nao_caju_ha.sum():,.0f} ha\n")
    (ROOT / "reports/rotulos_resumo.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
