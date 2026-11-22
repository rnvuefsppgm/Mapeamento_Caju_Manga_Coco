"""50 — Comparação do mapa de coco 2025 por município: área mapeada × PAM 2023–25 × Censo 2017 × MapBiomas 48
× Descals 2020 × rótulos; correlação; resumo da CV. Saída: v1/reports/comparacao_final.md (+ csv)
Uso: python 50_comparacao.py <lista>
"""
import json
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

from common import cfg, ext, ver

c = cfg()
lista = sys.argv[1]
pdir = ver(c, "predictions", "teste" if lista == "teste" else "2025", "x").parent
fon = ver(c, "data", "fontes", "x").parent
rot = ver(c, "data", "rotulos", "auto", "x").parent
rep = ver(c, "reports", "x").parent
regioes = [l.split() for l in ver(c, "data", "aoi", f"regions_{lista}.txt").read_text().split("\n") if l.strip()]
pc = pd.read_csv(ext(c, "pam", "pam_censo_coco.csv"), dtype={"cod": str})
mun = gpd.read_file(c["externos"]["municipios_shp"])
colc = next(k for k in mun.columns if k.upper().startswith("CD_MUN"))
L = []
for rid, crs in regioes:
    cod = rid.split("_")[0]
    with rasterio.open(pdir / f"{rid}_classe.tif") as s:
        cl, tr, shp = s.read(1), s.transform, s.shape
    mm = rasterize([(g, 1) for g in mun[mun[colc].astype(str) == cod].to_crs(crs).geometry], out_shape=shp, transform=tr, fill=0).astype(bool)
    mb, ds = (rasterio.open(fon / f"{rid}_{k}.tif").read(1) for k in ["mb", "descals"])
    lb = rasterio.open(rot / f"{rid}_rotulo.tif").read(1)
    r = pc[pc.cod == cod].iloc[0]
    L.append(dict(municipio=r.nome, mapa_ha=(cl[mm] == 1).sum() / 100, pam_ha=r.area_destinada_media,
                  censo_ha=r.area_existente_ha, mb48_ha=((mb == 48) & mm).sum() / 100, descals_ha=((ds == 1) & mm).sum() / 100,
                  rotulo_ha=((lb == 1) & mm).sum() / 100,
                  concord_rotulo=((cl == 1) & (lb == 1) & mm).sum() / max(1, ((lb == 1) & mm).sum())))
d = pd.DataFrame(L)
d["razao_pam"] = d.mapa_ha / d.pam_ha
d.round(2).to_csv(rep / "comparacao_final.csv", index=False)
cv = json.loads((rep / "cv_resumo.json").read_text()) if (rep / "cv_resumo.json").exists() else {}
tot = d[["mapa_ha", "pam_ha", "censo_ha", "mb48_ha", "descals_ha"]].sum()
lin = ["# Mapa de coco 2025 — comparação por município (v1)", "",
       f"CV espacial: IoU {cv.get('iou', float('nan')):.3f} (limiar {cv.get('limiar')}, trava {cv.get('trava_iou')}: "
       f"{'aprovado' if cv.get('aprovado') else 'REPROVADO — mapa preliminar'}); só ouro IoU {cv.get('so_ouro', {}).get('iou', float('nan')):.3f}.",
       f"Total: mapa {tot.mapa_ha:,.0f} ha | PAM {tot.pam_ha:,.0f} | Censo {tot.censo_ha:,.0f} | MapBiomas 48 {tot.mb48_ha:,.0f} | Descals {tot.descals_ha:,.0f}.",
       f"Correlação mapa × PAM: {np.corrcoef(d.mapa_ha, d.pam_ha)[0, 1]:.2f}; mapa × Censo: {np.corrcoef(d.mapa_ha, d.censo_ha.fillna(0))[0, 1]:.2f}. "
       f"Municípios com razão mapa/PAM entre 0,5 e 2,0: {int(d.razao_pam.between(0.5, 2).sum())} de {len(d)}.", "",
       "| Município | Mapa (ha) | PAM (ha) | Razão | Censo (ha) | MB 48 (ha) | Descals (ha) | Coco rotulado recuperado |",
       "|---|---|---|---|---|---|---|---|"]
lin += [f"| {r.municipio} | {r.mapa_ha:,.0f} | {r.pam_ha:,.0f} | {r.razao_pam:.2f} | {r.censo_ha:,.0f} | {r.mb48_ha:,.0f} | "
        f"{r.descals_ha:,.0f} | {100 * r.concord_rotulo:.0f}% |" for r in d.sort_values("pam_ha", ascending=False).itertuples()]
(rep / "comparacao_final.md").write_text("\n".join(lin), encoding="utf-8")
print("\n".join(lin[:5]))
