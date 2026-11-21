"""13 (v9) — Área de Lavoura Perene (MapBiomas Cobertura 10 m, classe 36, 2025) por município via API
da Plataforma MapBiomas, comparada com o mapa v8 e a PAM. Saída: reports/comparacao_perene_2025.md
"""
import json
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent
API = "https://plataforma.brasil.mapbiomas.org/api/v1/brazil"
H = {"tenant-id": "mapbiomas"}
MUN_CAT = 193  # categoria "município" (a 192 é outro tipo de território)


def tid(nome, geo):
    ts = requests.get(f"{API}/territories/search/{requests.utils.quote(nome)}", headers=H, timeout=60).json()["territories"]
    ts = [t for t in ts if str(t.get("geocode")) == geo and t.get("categoryId") == MUN_CAT]
    return ts[0]["id"] if ts else None


def perene(t, year=2025):
    q = {"territoryId": t, "subthemeKey": "coverage_coverage_10m", "legendKey": "coverage-10m-u5dful",
         "pixelValue": 36, "year": year}
    s = requests.get(f"{API}/statistics/area", headers=H, params=q, timeout=300).json()["statistic"]
    return sum(i["value"] for st in s for i in st["items"]) if s else 0.0


v8 = pd.DataFrame(json.loads((ROOT.parent / "v8" / "reports" / "area_caju_2025.json").read_text(encoding="utf-8")))
v8["cod"] = v8.regiao.str[1:8].astype(int)
pam = pd.read_csv(ROOT.parent / "dados_externos" / "ranking_caju_caatinga_2023_2025.csv")
t = v8.merge(pam[["cod", "nome", "area_ha_2025", "prod_med_t"]], on="cod")
t["nm"] = t.nome.str.split(" - ").str[0]
t["mb_tid"] = [tid(n, str(c)) for n, c in zip(t.nm, t.cod)]
t["perene_mb_2025"] = [perene(x) if x else None for x in t.mb_tid]
t = t.sort_values("prod_med_t", ascending=False)
f = lambda v: f"{v:,.0f}".replace(",", ".") if pd.notna(v) else "n/d"
q = pd.DataFrame({"Município": t.nome, "Mapa v8 (ha)": t.caju_ha.map(f),
                  "MapBiomas Lav. Perene 2025 (ha)": t.perene_mb_2025.map(f),
                  "PAM 2025 (ha)": t.area_ha_2025.map(f),
                  "Mapa/MB": (t.caju_ha / t.perene_mb_2025).round(2),
                  "Mapa/PAM": (t.caju_ha / t.area_ha_2025).round(2),
                  "MB/PAM": (t.perene_mb_2025 / t.area_ha_2025).round(2)})
tot = (f"Totais: mapa {f(t.caju_ha.sum())} ha | MapBiomas perene {f(t.perene_mb_2025.sum())} ha | "
       f"PAM {f(t.area_ha_2025.sum())} ha. Correlação entre municípios: mapa×MB "
       f"{t[['caju_ha','perene_mb_2025']].corr().iloc[0,1]:.2f}, MB×PAM {t[['perene_mb_2025','area_ha_2025']].corr().iloc[0,1]:.2f}, "
       f"mapa×PAM {t[['caju_ha','area_ha_2025']].corr().iloc[0,1]:.2f}.")
md = "# Caju 2025: mapa v8 × MapBiomas Lavoura Perene (10 m) × PAM\n\n" + q.to_markdown(index=False, disable_numparse=True) + "\n\n" + tot + "\n"
(ROOT / "reports" / "comparacao_perene_2025.md").write_text(md, encoding="utf-8")
t[["cod", "nome", "mb_tid", "perene_mb_2025"]].to_csv(ROOT / "reports" / "mapbiomas_perene_2025.csv", index=False)
print(md)
