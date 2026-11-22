"""00 — PAM/IBGE (tabela 1613) coco-da-baía 2023–2025 por município + ranking.

Saídas: dados_externos/pam/pam_coco_raw.json, pam_coco_municipios.csv,
        ranking_coco_{escopo}.csv. Retomável (não baixa de novo se o JSON existe).
"""
import json

import pandas as pd
import requests

from common import cfg, ext

c = cfg()
pc = c["pam"]
raw = ext(c, "pam", "pam_coco_raw.json")
if not raw.exists():
    vs = ",".join(str(v) for v in pc["variaveis"].values())
    anos = ",".join(str(a) for a in pc["anos"])
    url = (f"https://apisidra.ibge.gov.br/values/t/{pc['tabela']}/n6/all/v/{vs}"
           f"/p/{anos}/c82/{pc['produto']}/f/a")
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    raw.write_text(r.text, encoding="utf-8")

d = pd.DataFrame(json.loads(raw.read_text(encoding="utf-8"))[1:])
# Colunas SIDRA: D1C=município cód, D1N=nome, D2C=variável, D3C=ano, V=valor
d = d.rename(columns={"D1C": "cod", "D1N": "nome", "D2C": "var", "D3C": "ano", "V": "valor"})
d["valor"] = pd.to_numeric(d["valor"], errors="coerce")  # "-", "..." -> NaN
inv = {str(v): k for k, v in pc["variaveis"].items()}
d["var"] = d["var"].astype(str).map(inv)
w = d.pivot_table(index=["cod", "nome"], columns=["var", "ano"], values="valor", aggfunc="first")
w.columns = [f"{v}_{a}" for v, a in w.columns]
w = w.reset_index()
for v in pc["variaveis"]:
    cols = [f"{v}_{a}" for a in pc["anos"] if f"{v}_{a}" in w]
    w[f"{v}_media"] = w[cols].mean(axis=1)
w["uf"] = w["nome"].str[-2:]
w = w[w["area_destinada_media"] > 0].copy()
w.to_csv(ext(c, "pam", "pam_coco_municipios.csv"), index=False, encoding="utf-8-sig")

print(f"PAM coco: {len(w)} municípios com área > 0 | total área destinada média "
      f"{w['area_destinada_media'].sum():,.0f} ha")
print("Área por UF (ha, média 2023–25):",
      w.groupby("uf")["area_destinada_media"].sum().sort_values(ascending=False).head(10).round(0).to_dict())

for escopo in ["brasil", "nordeste"]:
    s = w if escopo == "brasil" else w[w["uf"].isin(c["recorte"]["ufs_nordeste"])]
    s = s.sort_values("area_destinada_media", ascending=False).copy()
    s["part_%"] = 100 * s["area_destinada_media"] / s["area_destinada_media"].sum()
    s["acum_%"] = s["part_%"].cumsum()
    s.to_csv(ext(c, "pam", f"ranking_coco_{escopo}.csv"), index=False, encoding="utf-8-sig")
    n = c["recorte"]["top_n"]
    print(f"[{escopo}] top {n} = {s['acum_%'].iloc[n - 1]:.1f}% da área; UFs: "
          f"{s.head(n)['uf'].value_counts().to_dict()}")
