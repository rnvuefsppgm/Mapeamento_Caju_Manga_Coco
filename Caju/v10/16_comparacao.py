"""16 (v10) — Comparação final: CV (C1 v7 × v10, mesmos folds e rótulos de campo) e área 2025
(v8 × v10 × MapBiomas Lavoura Perene × PAM) por município. Saída: reports/comparacao_final_v10.md
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
R = lambda n: json.loads((ROOT / "reports" / f"oof_{n}.json").read_text(encoding="utf-8"))
cv = pd.DataFrame([{"Modelo": k, **{x: round(v, 3) if isinstance(v, float) else v for x, v in R(n).items()}}
                   for k, n in [("C1 – U-Net v7 (rótulos originais)", "unet_emb_v7"), ("v10 – U-Net concordância", "unet_conc")]
                   if (ROOT / "reports" / f"oof_{n}.json").exists()])
a8 = pd.DataFrame(json.loads((ROOT.parent / "v8/reports/area_caju_2025.json").read_text(encoding="utf-8")))[["regiao", "caju_ha"]]
a10 = pd.DataFrame(json.loads((ROOT / "reports/area_caju_2025_v10.json").read_text(encoding="utf-8")))[["regiao", "caju_ha"]]
t = a8.merge(a10, on="regiao", suffixes=("_v8", "_v10"))
t["cod"] = t.regiao.str[1:8].astype(int)
p = pd.read_csv(ROOT.parent / "dados_externos/ranking_caju_caatinga_2023_2025.csv")[["cod", "nome", "area_ha_2025", "prod_med_t"]]
mb = pd.read_csv(ROOT.parent / "v9/reports/mapbiomas_perene_2025.csv")[["cod", "perene_mb_2025"]]
t = t.merge(p, on="cod").merge(mb, on="cod", how="left").sort_values("prod_med_t", ascending=False)
f = lambda v: f"{v:,.0f}".replace(",", ".") if pd.notna(v) else "n/d"
q = pd.DataFrame({"Município": t.nome, "v8 (ha)": t.caju_ha_v8.map(f), "v10 (ha)": t.caju_ha_v10.map(f),
                  "MapBiomas perene (ha)": t.perene_mb_2025.map(f), "PAM 2025 (ha)": t.area_ha_2025.map(f),
                  "v8/PAM": (t.caju_ha_v8 / t.area_ha_2025).round(2), "v10/PAM": (t.caju_ha_v10 / t.area_ha_2025).round(2)})
c = lambda a, b: t[[a, b]].corr().iloc[0, 1]
tot = (f"Totais: v8 {f(t.caju_ha_v8.sum())} | v10 {f(t.caju_ha_v10.sum())} | PAM {f(t.area_ha_2025.sum())} ha. "
       f"Correlação com a PAM: v8 {c('caju_ha_v8', 'area_ha_2025'):.2f} | v10 {c('caju_ha_v10', 'area_ha_2025'):.2f}. "
       f"Municípios com razão 0,5–2,5: v8 {((t.caju_ha_v8 / t.area_ha_2025).between(.5, 2.5)).sum()} | "
       f"v10 {((t.caju_ha_v10 / t.area_ha_2025).between(.5, 2.5)).sum()} de {len(t)}.")
md = ("# Comparação final (v10)\n\n## CV espacial — predições fora do fold, limiar global (rótulos de campo de Serra do Mel)\n\n"
      + cv.to_markdown(index=False) + "\n\n## Área 2025 por município\n\n" + q.to_markdown(index=False, disable_numparse=True)
      + "\n\n" + tot + "\n")
(ROOT / "reports/comparacao_final_v10.md").write_text(md, encoding="utf-8")
print(md)
