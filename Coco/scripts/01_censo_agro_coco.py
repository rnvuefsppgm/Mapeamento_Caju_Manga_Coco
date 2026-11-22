"""01 — Censo Agropecuário 2017 (SIDRA 6955), coco-da-baía (4945) por município.

Variáveis: 9504 estab. ≥ 50 pés · 10079 área existente (ha) · 10078 área colhida (ha) ·
10080 pés existentes ≥ 50 pés (mil) · 10082 pés existentes < 50 pés (mil).
Saída: dados_externos/censo_agro/censo2017_coco_municipios.csv. Retomável.
"""
import json

import pandas as pd
import requests

from common import cfg, ext

c = cfg()
ca = c["censo_agro"]
raw = ext(c, "censo_agro", "censo2017_coco_raw.json")
if not raw.exists():
    vs = ",".join(str(v) for v in ca["variaveis"].values())
    url = (f"https://apisidra.ibge.gov.br/values/t/{ca['tabela']}/n6/all/v/{vs}"
           f"/p/2017/c227/{ca['produto']}/f/a")
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    raw.write_text(r.text, encoding="utf-8")

d = pd.DataFrame(json.loads(raw.read_text(encoding="utf-8"))[1:])
d = d.rename(columns={"D1C": "cod", "D1N": "nome", "D2C": "var", "V": "valor"})
d["valor"] = pd.to_numeric(d["valor"], errors="coerce")  # "X" (sigilo), "-" -> NaN
inv = {str(v): k for k, v in ca["variaveis"].items()}
d["var"] = d["var"].astype(str).map(inv)
w = d.pivot_table(index=["cod", "nome"], columns="var", values="valor", aggfunc="first").reset_index()
w.columns.name = None
w["uf"] = w["nome"].str[-2:]
# densidade implícita (pés/ha) só onde há os dois valores — checa coerência com plantio (~160–205 pés/ha)
w["pes_por_ha"] = 1000 * w["pes_50mais_mil"] / w["area_existente_ha"]
w = w[(w["area_existente_ha"] > 0) | (w["pes_50mais_mil"] > 0)].copy()
w.to_csv(ext(c, "censo_agro", "censo2017_coco_municipios.csv"), index=False, encoding="utf-8-sig")

sig = d.groupby("var")["valor"].apply(lambda s: int(s.isna().sum())).to_dict()
print(f"Censo 2017 coco: {len(w)} municípios | área existente {w['area_existente_ha'].sum():,.0f} ha | "
      f"pés ≥50: {w['pes_50mais_mil'].sum():,.0f} mil")
print("Valores ausentes/sigilo por variável:", sig)
print("Área existente por UF (top 8):",
      w.groupby("uf")["area_existente_ha"].sum().sort_values(ascending=False).head(8).round(0).to_dict())
print("Pés/ha mediana:", round(w["pes_por_ha"].median(), 1))
