"""05 (v8) — Área de caju mapeada (2025) × área colhida PAM/IBGE (média 2023–2025 e 2025)."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
a = pd.DataFrame(json.loads((ROOT / "reports" / "area_caju_2025.json").read_text(encoding="utf-8")))
a = a[a.regiao.str.startswith("m")].copy()
a["cod"] = a.regiao.str[1:8].astype(int)
p = pd.read_csv(ROOT.parent / "dados_externos" / "ranking_caju_caatinga_2023_2025.csv")
t = a.merge(p[["cod", "nome", "area_med_ha", "area_ha_2025", "prod_med_t"]], on="cod", how="left")
t["mapa/PAM_2025"] = (t.caju_ha / t.area_ha_2025.replace(0, pd.NA)).round(2)
t = t.sort_values("prod_med_t", ascending=False)
cols = ["nome", "caju_ha", "area_ha_2025", "area_med_ha", "mapa/PAM_2025", "limiar"]
md = "# Área de caju 2025 — mapa (v8, unet_emb) × PAM\n\n" + t[cols].round(1).to_markdown(index=False)
md += (f"\n\nTotal mapeado: {t.caju_ha.sum():,.0f} ha | PAM 2025 (área colhida): "
       f"{t.area_ha_2025.sum():,.0f} ha.\n\nObs.: o mapa mede área plantada (inclui pomares jovens/"
       "não colhidos); a PAM mede área colhida declarada. Razões muito distantes de 1 indicam erro "
       "do modelo fora de Serra do Mel (treino só lá) ou diferença conceitual.\n")
(ROOT / "reports" / "comparacao_pam_2025.md").write_text(md, encoding="utf-8")
print(md)
