"""A5 — Tabela parcial/final: área do mapa automático × PAM 2025 × Censo 2017 para os municípios já processados.
Lê só dados/predicoes/xgb_auto/<rid>_{prob,conf}.tif existentes. Saída: stdout (<= 20 linhas) + reports/A5_parcial.md."""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

R = Path(__file__).resolve().parents[1]; D = R / "dados"; O = D / "predicoes" / "xgb_auto"
pam = pd.read_csv(D / "pam" / "ranking_pam_censo.csv").set_index("cod")
L = [l.split()[0] for l in (D / "aoi" / "regions_top15_caatinga.txt").read_text().split("\n") if l.strip()]
rows = []
for i, rid in enumerate(L, 1):
    if not (O / f"{rid}_conf.tif").exists():
        rows.append(f"| {i} | {rid[9:]} | … | … | … | … | {pam.loc[int(rid[1:8]), 'area_colhida_ha_2025']:,.0f} | … |")
        continue
    with rasterio.open(O / f"{rid}_prob.tif") as d:
        p = d.read(1); p = p[p != 255]
    with rasterio.open(O / f"{rid}_conf.tif") as d:
        c = (d.read(1) == 1).sum() / 100
    pa = pam.loc[int(rid[1:8]), "area_colhida_ha_2025"]; ce = pam.loc[int(rid[1:8]), "area_existente_ha"]
    p5, p8 = (p >= 50).sum() / 100, (p >= 80).sum() / 100
    rows.append(f"| {i} | {rid[9:]} | {p5:,.0f} | {p8:,.0f} | {c:,.0f} | {c / pa:.2f} | {pa:,.0f} | {ce:,.0f} |")
t = ["| # | Município | ha p≥0,5 | ha p≥0,8 | ha confiável | confiável/PAM | PAM 2025 (ha) | Censo 2017 (ha) |",
     "|---|---|---|---|---|---|---|---|"] + rows
(R / "reports" / "A5_parcial.md").write_text("# A5 — tabela parcial\n\n" + "\n".join(t) + "\n", encoding="utf-8")
print("\n".join(t))
