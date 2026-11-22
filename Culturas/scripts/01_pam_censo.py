"""01 — PAM/IBGE 2023–25 (média, tabela 1613) e Censo Agropecuário 2017 (tabela 6955) de caju, manga e coco
para TODOS os municípios mapeados (as 3 culturas em cada um). Saída: dados/pam_censo/pam_censo_3culturas.csv
Valores sigilosos (X) ou ausentes (-, ...) ficam vazios. Retomável (pula se o csv existe; --refazer para baixar de novo).
"""
import sys

import pandas as pd
import requests

from common import cfg, municipios_por_cultura, p

c = cfg()
out = p("dados", "pam_censo", "pam_censo_3culturas.csv")
if out.exists() and "--refazer" not in sys.argv:
    print(f"já existe: {out}"); sys.exit()
cods = ",".join(str(x) for x in sorted(set().union(*municipios_por_cultura(c).values())))


def sidra(tab, vs, per, cls, prod):
    u = f"https://apisidra.ibge.gov.br/values/t/{tab}/n6/{cods}/v/{vs}/p/{per}/c{cls}/{prod}/f/a"
    d = pd.DataFrame(requests.get(u, timeout=600).json()[1:])
    d["valor"] = pd.to_numeric(d["V"], errors="coerce")
    return d.rename(columns={"D1C": "cod", "D1N": "nome", "D2C": "var", "D3C": "ano"})[["cod", "nome", "var", "ano", "valor"]]


L = []
pm, ce = c["pam"], c["censo"]
for cult, v in c["culturas"].items():
    a = sidra(pm["tabela"], ",".join(map(str, pm["variaveis"].values())), ",".join(map(str, pm["anos"])), 82, v["pam_produto"])
    a = a.groupby(["cod", "nome", "var"]).valor.mean().unstack()   # média dos anos (ausente não entra)
    inv = {str(k2): k for k, k2 in pm["variaveis"].items()}
    a = a.rename(columns=inv).add_prefix("pam_").add_suffix("_ha")
    b = sidra(ce["tabela"], ",".join(map(str, ce["variaveis"].values())), 2017, 227, v["censo_produto"])
    b = b.pivot_table(index=["cod", "nome"], columns="var", values="valor")
    b = b.rename(columns={str(k2): k for k, k2 in ce["variaveis"].items()}).add_prefix("censo_")
    d = a.join(b, how="outer").reset_index()
    d["cultura"] = cult
    L.append(d)
    print(f"{cult}: PAM área destinada {d.pam_area_destinada_ha.sum():,.0f} ha | Censo área existente "
          f"{d.censo_area_existente_ha.sum():,.0f} ha ({len(d)} municípios)")
pd.concat(L).to_csv(out, index=False)
print(f"ok -> {out}")
