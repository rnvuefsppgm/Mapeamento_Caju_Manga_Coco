"""A1 — Censo Agropecuário 2017 (SIDRA 6955, c227 Manga = 4968) por município + cruzamento com PAM 2023-25 e bioma.

Adaptado de Mapa_COCO/scripts/01_censo_agro_coco.py. Retomável (não baixa de novo).
Saídas: dados/censo_agro/censo2017_manga_{raw.json,municipios.csv}, dados/pam/ranking_pam_censo.csv,
        reports/A1_ranking_manga.md (15 maiores da Caatinga). Imprime <= 20 linhas.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import requests

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.config import load_config, resolve_path  # noqa: E402

cfg = load_config(RAIZ / "config/config.yaml")
ca = cfg["manga"]["censo_agro"]
out = resolve_path(cfg, "censo")
raw = out / "censo2017_manga_raw.json"
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
w["cod"] = w["cod"].astype(int)
w["pes_por_ha"] = 1000 * w["pes_50mais_mil"] / w["area_existente_ha"]
w = w[(w["area_existente_ha"] > 0) | (w["pes_50mais_mil"] > 0)].copy()
w.to_csv(out / "censo2017_manga_municipios.csv", index=False, encoding="utf-8-sig")

# cruzamento com a PAM (02_pam_municipios.py) e bioma
pam = pd.read_csv(resolve_path(cfg, "pam") / "pam_manga_2023_2025.csv")
x = pam.merge(w.drop(columns="nome"), on="cod", how="outer")
x["nome"] = x["nome"].fillna(x["cod"].map(w.set_index("cod")["nome"]))
x["razao_pam_censo"] = x["area_colhida_ha_2025"] / x["area_existente_ha"]
x["alerta"] = (x["razao_pam_censo"] > 2) | (x["razao_pam_censo"] < 0.5)
x = x.sort_values("prod_med_t", ascending=False)
x.to_csv(resolve_path(cfg, "pam") / "ranking_pam_censo.csv", index=False, encoding="utf-8-sig")
ca_ = x[x["bioma"] == cfg["manga"]["selecao"]["bioma"]].head(15)
tot = x.loc[x["bioma"] == "Caatinga", "prod_med_t"].sum()
lin = ["| # | Município | Prod. média PAM 2023-25 (t) | Área colhida PAM 2025 (ha) | Área existente Censo 2017 (ha) | "
       "Estab. ≥50 pés | Pés/ha | PAM/Censo | Alerta |", "|---|---|---|---|---|---|---|---|---|"]
for i, r in enumerate(ca_.itertuples(), 1):
    lin.append(f"| {i} | {r.nome} | {r.prod_med_t:,.0f} | {r.area_colhida_ha_2025:,.0f} | {r.area_existente_ha:,.0f} | "
               f"{r.estab_50mais:,.0f} | {r.pes_por_ha:,.0f} | {r.razao_pam_censo:.2f} | {'sim' if r.alerta else ''} |")
(resolve_path(cfg, "reports") / "A1_ranking_manga.md").write_text(
    "# A1 — 15 maiores produtores de manga da Caatinga (PAM × Censo 2017)\n\n"
    f"Os 15 somam {100 * ca_['prod_med_t'].sum() / tot:.1f}% da produção da Caatinga. "
    "Alerta = área colhida PAM 2025 fora de 0,5–2× a área existente no Censo 2017.\n\n" + "\n".join(lin) + "\n",
    encoding="utf-8")
sig = d.groupby("var")["valor"].apply(lambda s: int(s.isna().sum())).to_dict()
print(f"Censo 2017 manga: {len(w)} municípios | área existente {w['area_existente_ha'].sum():,.0f} ha | "
      f"pés ≥50: {w['pes_50mais_mil'].sum():,.0f} mil | pés/ha mediana {w['pes_por_ha'].median():.0f}")
print("Ausentes/sigilo por variável:", sig)
print(f"Top 15 Caatinga = {100 * ca_['prod_med_t'].sum() / tot:.1f}% da produção do bioma | alertas: {int(ca_['alerta'].sum())}")
print("Tabela: reports/A1_ranking_manga.md")
