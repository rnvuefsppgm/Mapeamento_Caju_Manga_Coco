"""14 — Rótulos por RANKING com prior de área (receita caju v10a/v10b), a partir de <r>_xgb.tif do 13.

Por município (ref = max(PAM área destinada 2023–25, Censo 2017 área existente)):
  candidatos = válidos, fora de MB segura e de caju ≥ 50
  coco  = candidatos no topo do ranking de prob. até 0,6 × ref, com prob ≥ 35; manchas ≥ 1 ha
  não   = fora das classes compatíveis C (MapBiomas) | caju ≥ 80 | (fora do topo 3 × ref e prob ≤ 60)
  resto = ignorado (255); anel de 2 px em volta do coco = ignorado; R-ouro/prata sobrepõe.
Rápido (sem embeddings). Saídas: v1/data/rotulos/auto/<r>_rotulo.tif, v1/reports/rotulos_auto.md
Uso: python 14_rotulos_regras.py <lista>
"""
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy import ndimage

from common import cfg, ext, ver

c = cfg()
R = c["rotulos"]
lista = sys.argv[1]
TOPO_POS, TOPO_NEG, P_MIN, P_NEG = 0.6, 3.0, 35, 60
fon = ver(c, "data", "fontes", "x").parent
rot = ver(c, "data", "rotulos", "x").parent
regioes = [l.split() for l in ver(c, "data", "aoi", f"regions_{lista}.txt").read_text().split("\n") if l.strip()]
ouro = gpd.read_file(rot / "r_ouro.gpkg")
ouro = ouro[ouro.nivel == "ouro"]   # v1b: prata fora (QC 21/11)
COCO_MB = R.get("classes_coco_mb", [48, 21, 41])
ERO = R.get("erosao_coco_px", 2)
pc = pd.read_csv(ext(c, "pam", "pam_censo_coco.csv"), dtype={"cod": str})
mun = gpd.read_file(c["externos"]["municipios_shp"])
colc = next(k for k in mun.columns if k.upper().startswith("CD_MUN"))


def corte(v, n):  # valor da prob. na posição n do ranking decrescente
    return np.partition(v, len(v) - n)[len(v) - n] if 0 < n < len(v) else (v.min() if n >= len(v) else 101)


linhas = []
for rid, crs in regioes:
    with rasterio.open(fon / f"{rid}_xgb.tif") as s:
        pr, prof, tr, shp = s.read(1), s.profile, s.transform, s.shape
    mb, cj = (rasterio.open(fon / f"{rid}_{k}.tif").read(1) for k in ["mb", "caju"])
    valid = np.zeros(shp, bool)   # validade = cobertura REAL dos embeddings (não MapBiomas) — bug corrigido 21/11
    for t in sorted((ver(c, "data", "embeddings", "2025", rid, "x").parent).glob("*.tif")):
        with rasterio.open(t) as e:
            r0, c0 = round((tr.f - e.bounds.top) / 10), round((e.bounds.left - tr.c) / 10)
            valid[r0:r0 + e.height, c0:c0 + e.width] |= e.read(1) != 0
    seg = np.isin(mb, R["classes_seguras_nao_coco_mb"])
    C = np.isin(mb, R["classes_compativeis_mb"])
    cjv = cj != 255
    cand = valid & ~seg & ~(cjv & (cj >= 50))
    mm = rasterize([(g, 1) for g in mun[mun[colc].astype(str) == rid.split("_")[0]].to_crs(crs).geometry],
                   out_shape=shp, transform=tr, fill=0).astype(bool)
    row = pc[pc.cod == rid.split("_")[0]]
    ref = float(np.nanmax([row.area_destinada_media.iloc[0], row.area_existente_ha.iloc[0]]))
    v = pr[cand & mm & np.isin(mb, COCO_MB)]
    t_pos = max(P_MIN, corte(v, int(TOPO_POS * ref * 100)))
    t_neg = corte(v, int(TOPO_NEG * ref * 100))
    coco = cand & (pr >= t_pos) & np.isin(mb, COCO_MB)
    nucleo = ndimage.binary_erosion(coco, iterations=ERO)   # tira bordas/estradas internas (QC: 11 erros de borda)
    borda = coco & ~nucleo
    coco = nucleo
    lab, _ = ndimage.label(coco)
    coco &= (np.bincount(lab.ravel()) >= R["mancha_min_ha"] * 100)[lab]
    # empate no topo (prob. saturada em 100 no SF): sorteia manchas até TOPO_POS × ref; o resto vira ignorado
    excesso = np.zeros(shp, bool)
    if (coco & mm).sum() > TOPO_POS * ref * 100 * 1.2:
        lab, n = ndimage.label(coco)
        area_mm = np.bincount(lab[mm], minlength=n + 1)
        ordem = np.random.default_rng(42).permutation(np.arange(1, n + 1))
        manter, acum = [], 0
        for k in ordem:  # guloso: inclui a mancha se couber no teto (não para na primeira grande)
            if acum + area_mm[k] <= TOPO_POS * ref * 100:
                manter.append(k); acum += area_mm[k]
        excesso = coco & ~np.isin(lab, manter)
        coco &= ~excesso
    nao = valid & (~C | (cjv & (cj >= 80)) | (cand & (pr < t_neg) & (pr <= P_NEG)))
    out = np.full(shp, 255, np.uint8)
    out[nao] = 0
    out[coco] = 1
    out[excesso] = 255
    out[borda] = 255
    out[ndimage.binary_dilation(coco, iterations=R["anel_ignorado_px"]) & ~coco] = 255
    o = ouro.to_crs(crs)
    for cl, val in [("nao_coco", 0), ("coco", 1)]:
        g = [x for x in o[o.classe == cl].geometry if not x.is_empty]
        if g:
            out[rasterize([(x, 1) for x in g], out_shape=shp, transform=tr, fill=0).astype(bool)] = val
    out[~valid] = 255
    with rasterio.open(rot / "auto" / f"{rid}_rotulo.tif", "w", **prof) as d:
        d.write(out, 1)
    ha = lambda a: (a & mm).sum() / 100
    linhas.append(f"| {rid} | {ha(out == 1):,.0f} | {ref:,.0f} | {ha(out == 1) / ref:.2f} | {t_pos} | "
                  f"{ha(out == 0):,.0f} | {100 * ha(out == 255) / max(1, mm.sum() / 100):.0f}% |")
    print(linhas[-1], flush=True)

ver(c, "reports", "rotulos_auto.md").write_text("\n".join([
    "# Rótulos automáticos v1 (ranking com prior de área)", "",
    f"coco = topo {TOPO_POS}×ref com prob ≥ {P_MIN}; não = fora de C | caju ≥ 80 | fora do topo {TOPO_NEG}×ref e prob ≤ {P_NEG}.",
    "ref = max(PAM área destinada 2023–25, Censo 2017 área existente). Áreas dentro do município.", "",
    "| Região | Coco (ha) | Ref. (ha) | Razão | Limiar prob. coco | Não-coco (ha) | Ignorado |",
    "|---|---|---|---|---|---|---|", *linhas]), encoding="utf-8")
