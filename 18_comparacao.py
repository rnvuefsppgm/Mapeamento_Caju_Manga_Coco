"""18 (v1) — Comparação por município: mapa v1 × PAM 2025 × Censo 2017 × mapa automático A5 (confiável).
Região "confiável" = Chapada (paisagem das amostras); demais = "extrapolação" (valores provisórios).
Saída: reports/v1_comparacao.md (+ <= 20 linhas no terminal)."""
import json
from pathlib import Path

import pandas as pd
import rasterio
import yaml

R = Path(__file__).parent; D = R / "dados"
CF = yaml.safe_load(open(R / "config/config.yaml", encoding="utf-8"))
CON = set(CF["amostragem"]["auto_classificacao"]["regiao_conhecida"])
pam = pd.read_csv(D / "pam/ranking_pam_censo.csv").set_index("cod")
t = pd.DataFrame(json.loads((R / "reports/v1_area.json").read_text(encoding="utf-8")))
t["cod"] = t.regiao.str[1:8].astype(int)
t["municipio"] = t.cod.map(pam["nome"])
t["pam_2025_ha"] = t.cod.map(pam["area_colhida_ha_2025"])
t["censo_2017_ha"] = t.cod.map(pam["area_existente_ha"])
a5 = []
for r in t.regiao:
    f = D / f"predicoes/xgb_auto/{r}_conf.tif"
    a5.append((rasterio.open(f).read(1) == 1).sum() / 100 if f.exists() else float("nan"))
t["a5_confiavel_ha"] = a5
t["mapa/PAM"] = t.manga_ha / t.pam_2025_ha
t["tipo"] = ["confiável" if c in CON else "extrapolação" for c in t.cod]
t = t.sort_values("pam_2025_ha", ascending=False)
cols = ["municipio", "tipo", "manga_ha", "pam_2025_ha", "censo_2017_ha", "mapa/PAM", "a5_confiavel_ha", "manga_ha_sem_mascara"]
tot = t[["manga_ha", "pam_2025_ha"]].sum()
corr = t[["manga_ha", "pam_2025_ha"]].corr().iloc[0, 1]
cab = (f"Total mapa {tot.manga_ha:,.0f} ha x PAM {tot.pam_2025_ha:,.0f} ha (razão {tot.manga_ha / tot.pam_2025_ha:.2f}); "
       f"correlação com a PAM {corr:.2f}. Municípios 'extrapolação' estão fora da paisagem das amostras: valores provisórios.")
(R / "reports/v1_comparacao.md").write_text("# Manga 2025 — v1 (rede treinada só com amostras da Chapada)\n\n" + cab + "\n\n"
                                            + t[cols].round(2).to_markdown(index=False) + "\n", encoding="utf-8")
print(cab)
print(t[cols].round(2).head(15).to_string(index=False))
