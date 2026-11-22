"""A5 — Mapa AUTOMÁTICO de manga por pixel nos municípios da lista, com o classificador XGBoost da A4a (GPU).

Para cada bloco de 1024 px do VRT: p_manga calibrada (isotônica da A4a), p_caju, p_coco. Pixel confiável = regra da
opção A (p >= 0,80 na Chapada; p >= 0,90 e margem >= 0,5 sobre caju/coco nas regiões novas), fora das classes seguras
MapBiomas e dentro de classe compatível; abertura 1 px; manchas >= 1 ha. Compara a área com a PAM 2025 e o Censo 2017.
Saídas: dados/predicoes/xgb_auto/<rid>_prob.tif (0-100; 255 = fora), <rid>_conf.tif (0/1; 255 fora),
        dados/predicoes/xgb_auto/manchas_confiaveis.gpkg, reports/A5_mapa_xgb.md. Retomável por município. <= 20 linhas.
Uso: python A5_mapa_xgb.py [lista]
"""
import sys
from pathlib import Path

import cupy as cp
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xgboost as xgb
import yaml
from rasterio.features import rasterize, shapes
from rasterio.windows import Window
from scipy import ndimage
from shapely.geometry import shape

RAIZ = Path(__file__).resolve().parents[1]
CF = yaml.safe_load(open(RAIZ / "config/config.yaml", encoding="utf-8"))
AC = CF["amostragem"]["auto_classificacao"]; MX = CF["amostragem"]["mapa_xgb"]
SEG = CF["amostragem"]["rotulos"]["classes_seguras_nao_coco_mb"]
D = RAIZ / "dados"; OUT = D / "predicoes" / "xgb_auto"; OUT.mkdir(parents=True, exist_ok=True)
lista = sys.argv[1] if len(sys.argv) > 1 else "top15_caatinga"
regioes = [l.split()[0] for l in (D / "aoi" / f"regions_{lista}.txt").read_text().split("\n") if l.strip()]
bst = xgb.Booster(); bst.load_model(RAIZ / "modelos/a4a/xgb_4classes.json"); bst.set_param({"device": "cuda"})
z = np.load(RAIZ / "modelos/a4a/isotonica.npz"); IX, IY = cp.asarray(z["x"]), cp.asarray(z["y"])
mun = gpd.read_file(D / "pam" / "municipios_selecionados.gpkg").to_crs(CF["manga"]["crs"]).set_index("cod")
B = MX["bloco_px"]
PROF = dict(driver="GTiff", count=1, dtype="uint8", nodata=255, tiled=True, blockxsize=512, blockysize=512, compress="deflate")

for rid in regioes:
    fp, fc = OUT / f"{rid}_prob.tif", OUT / f"{rid}_conf.tif"
    if fc.exists():
        continue
    cod = int(rid[1:8]); conhecida = cod in AC["regiao_conhecida"]
    with rasterio.open(D / "embeddings" / f"{rid}.vrt") as e, rasterio.open(D / "mapbiomas" / f"{rid}_mb.tif") as m:
        H, W, tr = e.height, e.width, e.transform
        dentro = rasterize([(mun.loc[cod].geometry, 1)], (H, W), transform=tr, dtype="uint8").astype(bool)
        Pm = np.full((H, W), 255, np.uint8); Cf = np.zeros((H, W), bool)
        for r0 in range(0, H, B):
            for c0 in range(0, W, B):
                w = Window(c0, r0, min(B, W - c0), min(B, H - r0))
                din = dentro[r0:r0 + w.height, c0:c0 + w.width]
                if not din.any():
                    continue
                E = e.read(window=w); mb = m.read(1, window=w)
                ok = din & (E[0] != 0)
                if not ok.any():
                    continue
                x = (cp.asarray(E[:, ok].T).astype(cp.float32) - 128.0) / 127.0
                P = bst.inplace_predict(x)                                # (n,4) na GPU
                pm = cp.interp(cp.ascontiguousarray(P[:, 1]), IX, IY)
                cf = pm >= AC["p_ok"] if conhecida else (pm >= MX["p_nova"]) & (pm - cp.maximum(P[:, 2], P[:, 3]) >= MX["margem_caju_coco"])
                pm_h, cf_h = cp.asnumpy(pm), cp.asnumpy(cf)
                blk = np.full(ok.shape, 255, np.uint8); blk[ok] = np.round(pm_h * 100).astype(np.uint8)
                Pm[r0:r0 + w.height, c0:c0 + w.width] = blk
                cb = np.zeros(ok.shape, bool); cb[ok] = cf_h
                cb &= ~np.isin(mb, SEG) & np.isin(mb, AC["classes_compativeis"])
                Cf[r0:r0 + w.height, c0:c0 + w.width] = cb
                del x, P, pm, cf
        Cf = ndimage.binary_opening(Cf, iterations=MX["abertura_px"])
        lab, n = ndimage.label(Cf)
        if n:
            tam = ndimage.sum(Cf, lab, range(1, n + 1))
            Cf = np.isin(lab, np.nonzero(tam >= MX["mancha_min_ha"] * 100)[0] + 1)
        C8 = np.where(dentro, Cf.astype(np.uint8), 255).astype(np.uint8)
        for f, a in [(fp, Pm), (fc, C8)]:
            with rasterio.open(f, "w", height=H, width=W, crs=e.crs, transform=tr, **PROF) as d:
                d.write(a, 1)
        mb_all = None
    print(f"{rid}: {Cf.sum() / 100:,.0f} ha confiáveis", flush=True)

# ---------------- resumo ----------------
pam = pd.read_csv(D / "pam" / "ranking_pam_censo.csv").set_index("cod")
rows, manchas = [], []
for rid in regioes:
    cod = int(rid[1:8])
    with rasterio.open(OUT / f"{rid}_prob.tif") as d:
        Pm = d.read(1)
    with rasterio.open(OUT / f"{rid}_conf.tif") as d:
        C = d.read(1) == 1; tr, crs = d.transform, d.crs
    v = Pm[Pm != 255]
    lab, n = ndimage.label(C)
    if n:
        for g, val in shapes(lab.astype(np.int32), mask=C, transform=tr):
            manchas.append({"regiao": rid, "geometry": shape(g)})
    rows.append({"regiao": rid, "conhecida": cod in AC["regiao_conhecida"],
                 "ha_p50": (v >= 50).sum() / 100, "ha_p80": (v >= 80).sum() / 100, "ha_confiavel": C.sum() / 100,
                 "n_manchas": n, "pam_2025_ha": pam.loc[cod, "area_colhida_ha_2025"],
                 "censo_2017_ha": pam.loc[cod, "area_existente_ha"]})
t = pd.DataFrame(rows)
t["conf/PAM"] = t.ha_confiavel / t.pam_2025_ha
t["p50/PAM"] = t.ha_p50 / t.pam_2025_ha
g = gpd.GeoDataFrame(manchas, crs=CF["manga"]["crs"])
if len(g):
    g["area_ha"] = g.area / 1e4
    g.to_file(OUT / "manchas_confiaveis.gpkg", driver="GPKG")
z = open(RAIZ / "modelos/a4a/cv.txt", encoding="utf-8").readline().strip()
(RAIZ / "reports" / "A5_mapa_xgb.md").write_text("\n".join([
    "# A5 — Mapa automático de manga por pixel (XGBoost da A4a) — rótulo AUTOMÁTICO", "",
    f"Modelo: `modelos/a4a/xgb_4classes.json` ({z}). Pixel confiável = regra da opção A; fora das classes seguras e dentro "
    f"das compatíveis MapBiomas; abertura {MX['abertura_px']} px; manchas ≥ {MX['mancha_min_ha']} ha. Área dentro do limite municipal (sem buffer).", "",
    t.round(2).to_markdown(index=False), "",
    f"Total confiável: {t.ha_confiavel.sum():,.0f} ha · PAM 2025 dos 15: {t.pam_2025_ha.sum():,.0f} ha · "
    f"p ≥ 0,5: {t.ha_p50.sum():,.0f} ha.", "",
    "Rasters: `dados/predicoes/xgb_auto/<regiao>_prob.tif` (0–100) e `_conf.tif`; manchas: `manchas_confiaveis.gpkg`."]), encoding="utf-8")
print(f"Total confiável {t.ha_confiavel.sum():,.0f} ha | p>=0,5 {t.ha_p50.sum():,.0f} ha | PAM 2025 {t.pam_2025_ha.sum():,.0f} ha")
print("Relatório: reports/A5_mapa_xgb.md")
