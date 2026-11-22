"""16 (v10a) — Comparação: CV (C1 × v10 × v10a, rótulos de campo de Serra do Mel) e área 2025
(v8 × v10 × v10a × MapBiomas Lavoura Perene × PAM). Saída: reports/comparacao_final_v10a.md
Atenção: a PAM entrou como prior de QUANTIDADE de rótulos na v10a, então a comparação com a PAM é otimista.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
V10 = ROOT.parent / "v10"
J = lambda p: json.loads(p.read_text(encoding="utf-8"))
cv = pd.DataFrame([{"Modelo": k, **{x: round(v, 3) if isinstance(v, float) else v for x, v in J(p).items()}}
                   for k, p in [("C1 – U-Net v7", V10 / "reports/oof_unet_emb_v7.json"),
                                ("v10 – concordância", V10 / "reports/oof_unet_conc.json"),
                                ("v10a – XGB regional + ranking", ROOT / "reports/oof_unet_conc_a.json")] if p.exists()])
A = lambda p: pd.DataFrame(J(p))[["regiao", "caju_ha"]]
t = (A(ROOT.parent / "v8/reports/area_caju_2025.json").rename(columns={"caju_ha": "v8"})
     .merge(A(V10 / "reports/area_caju_2025_v10.json").rename(columns={"caju_ha": "v10"}), on="regiao")
     .merge(A(ROOT / "reports/area_caju_2025_v10.json").rename(columns={"caju_ha": "v10a"}), on="regiao"))
t["cod"] = t.regiao.str[1:8].astype(int)
p = pd.read_csv(ROOT.parent / "dados_externos/ranking_caju_caatinga_2023_2025.csv")[["cod", "nome", "area_ha_2025", "prod_med_t"]]
mb = pd.read_csv(ROOT.parent / "v9/reports/mapbiomas_perene_2025.csv")[["cod", "perene_mb_2025"]]
t = t.merge(p, on="cod").merge(mb, on="cod", how="left").sort_values("prod_med_t", ascending=False)
f = lambda v: f"{v:,.0f}".replace(",", ".") if pd.notna(v) else "n/d"
q = pd.DataFrame({"Município": t.nome, "v8 (ha)": t.v8.map(f), "v10 (ha)": t.v10.map(f), "v10a (ha)": t.v10a.map(f),
                  "MapBiomas perene (ha)": t.perene_mb_2025.map(f), "PAM 2025 (ha)": t.area_ha_2025.map(f),
                  "v10/PAM": (t.v10 / t.area_ha_2025).round(2), "v10a/PAM": (t.v10a / t.area_ha_2025).round(2)})
c = lambda a: t[[a, "area_ha_2025"]].corr().iloc[0, 1]
ok = lambda a: int((t[a] / t.area_ha_2025).between(.5, 2.5).sum())
tot = "\n".join(f"- {a}: total {f(t[a].sum())} ha | correlação com a PAM {c(a):.2f} | razão 0,5–2,5 em {ok(a)} de {len(t)}"
                for a in ("v8", "v10", "v10a")) + f"\n- PAM: {f(t.area_ha_2025.sum())} ha"
md = ("# Comparação final (v10a)\n\n## CV espacial — fora do fold, limiar global (campo de Serra do Mel)\n\n"
      + cv.to_markdown(index=False) + "\n\n## Área 2025 por município\n\n" + q.to_markdown(index=False, disable_numparse=True)
      + "\n\n" + tot + "\n\n*A PAM definiu a quantidade de rótulos na v10a; a comparação com a PAM é otimista.*\n")
(ROOT / "reports/comparacao_final_v10a.md").write_text(md, encoding="utf-8")
print(md)
