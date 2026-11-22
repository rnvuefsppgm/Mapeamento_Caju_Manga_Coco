"""10 — Amostra de candidatos por faixa para conferência visual (folhas do 07).

Uso: python 10_qc_candidatos.py <tag_qc> <gpkg_candidatos> [n_por_faixa=6]
Gera grupos/<tag_qc>_reps.csv + <tag_qc>_manchas.gpkg e chama o 07 (z17). A decisão visual vai em
v1/data/rotulos/qc/qc_decisoes.csv (id_qc, tag, cand_id, regiao, faixa, classe ∈ coco|nao_coco|duvida, nota).
"""
import subprocess
import sys

import geopandas as gpd
import pandas as pd

from common import cfg, ver

c = cfg()
tag, gp = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
d = ver(c, "data", "rotulos", "grupos", "x").parent
g = gpd.read_file(gp)
s = (g.groupby("faixa", group_keys=False).apply(lambda x: x.sample(min(n, len(x)), random_state=7))
     .sort_values(["faixa", "regiao"]))
p = s.geometry.representative_point()
r = pd.DataFrame({"id": [f"{tag}_{i:02d}" for i in range(len(s))], "cand_id": s["id"].values, "grupo": 0,
                  "regiao": s["regiao"].values, "lon": p.x.values, "lat": p.y.values,
                  "area_ha": s["area_ha"].round(1).values, "mb": s["mb_moda"].values, "faixa": s["faixa"].values})
r.to_csv(d / f"{tag}_reps.csv", index=False)
s.assign(regiao=s["regiao"])[["regiao", "geometry"]].to_file(d / f"{tag}_manchas.gpkg", driver="GPKG")
print(r[["id", "cand_id", "regiao", "faixa", "area_ha"]].to_string(index=False))
subprocess.run([sys.executable, "-W", "ignore", __file__.replace("10_qc_candidatos.py", "07_folhas_visuais.py"),
                tag, "17", ",".join(r["id"])], check=True)
