"""33 (v10b) — Versão SÓ U-Net (sem fusão com o XGBoost): classe = prob U-Net >= 0,40 (limiar do CV fora do fold),
mesmo pós-processamento de 1 ha. Usa os _prob.tif já salvos pelo 15 (não prevê de novo).
Saídas: predictions/2025/<rid>_class_unet.tif, predictions/caju_2025_25mun_classe_unet.tif,
reports/comparacao_unet_x_fusao.md (com fusão × só U-Net × MapBiomas perene × PAM).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge

from src import pred_a as P

ROOT = Path(__file__).parent
OD = ROOT / "predictions/2025"
THR = 40  # limiar do CV v10b (reports/oof_unet_conc_b.json)
rows = []
for f in sorted(OD.glob("m*_prob.tif")):
    rid = f.name.replace("_prob.tif", "")
    with rasterio.open(f) as d:
        U, prof = d.read(1), d.profile
    nod = U == 255
    cls = P.pos_proc((U >= THR) & ~nod) & ~nod
    prof.update(nodata=255, compress="deflate")
    with rasterio.open(OD / f"{rid}_class_unet.tif", "w", **prof) as d:
        d.write(np.where(nod, 255, cls).astype("uint8"), 1)
    rows.append({"regiao": rid, "so_unet_ha": round(cls.sum() / 100, 1)})
    print(rid, rows[-1]["so_unet_ha"], flush=True)
srcs = [rasterio.open(OD / f"{r['regiao']}_class_unet.tif") for r in rows]
arr, tr = merge(srcs, nodata=255, method="first")
pr = dict(driver="COG", height=arr.shape[1], width=arr.shape[2], count=1, dtype="uint8", crs=srcs[0].crs, transform=tr,
          nodata=255, compress="DEFLATE", predictor=2, overviews="AUTO", BIGTIFF="IF_SAFER")
with rasterio.open(ROOT / "predictions/caju_2025_25mun_classe_unet.tif", "w", **pr) as d:
    d.write(arr[0], 1)
for s in srcs:
    s.close()
t = pd.DataFrame(rows).merge(pd.DataFrame(json.loads((ROOT / "reports/area_caju_2025_v10.json").read_text(encoding="utf-8")))
                             [["regiao", "caju_ha"]].rename(columns={"caju_ha": "fusao_ha"}), on="regiao")
t["cod"] = t.regiao.str[1:8].astype(int)
p = pd.read_csv(ROOT.parent / "dados_externos/ranking_caju_caatinga_2023_2025.csv")[["cod", "nome", "area_ha_2025", "prod_med_t"]]
mb = pd.read_csv(ROOT.parent / "v9/reports/mapbiomas_perene_2025.csv")[["cod", "perene_mb_2025"]]
v10 = pd.DataFrame(json.loads((ROOT.parent / "v10/reports/area_caju_2025_v10.json").read_text(encoding="utf-8")))[["regiao", "caju_ha"]]
t = t.merge(v10.rename(columns={"caju_ha": "v10_ha"}), on="regiao").merge(p, on="cod").merge(mb, on="cod", how="left")
t = t.sort_values("prod_med_t", ascending=False)
fm = lambda v: f"{v:,.0f}".replace(",", ".") if pd.notna(v) else "n/d"
q = pd.DataFrame({"Município": t.nome, "PAM 2025 (ha)": t.area_ha_2025.map(fm), "v10 (ha)": t.v10_ha.map(fm),
                  "v10b só U-Net (ha)": t.so_unet_ha.map(fm), "v10b fusão (ha)": t.fusao_ha.map(fm),
                  "MapBiomas perene (ha)": t.perene_mb_2025.map(fm), "v10/PAM": (t.v10_ha / t.area_ha_2025).round(2),
                  "só U-Net/PAM": (t.so_unet_ha / t.area_ha_2025).round(2), "fusão/PAM": (t.fusao_ha / t.area_ha_2025).round(2)})
c = lambda a: t[[a, "area_ha_2025"]].corr().iloc[0, 1]
ok = lambda a: int((t[a] / t.area_ha_2025).between(.5, 2.5).sum())
mae = lambda a: float((np.log(t[a].clip(lower=1) / t.area_ha_2025)).abs().median())
res = "\n".join(f"- **{n}**: total {fm(t[a].sum())} ha | correlação com a PAM {c(a):.2f} | razão 0,5–2,5 em {ok(a)} de {len(t)} | "
                f"erro típico (mediana de |log razão|) {mae(a):.2f}"
                for n, a in (("v10", "v10_ha"), ("v10b só U-Net", "so_unet_ha"), ("v10b fusão", "fusao_ha")))
md = ("# v10b — só U-Net × fusão × PAM (2025)\n\nSó U-Net: prob ≥ 0,40 (limiar do CV); fusão: 0,6·U-Net + 0,4·XGB regional ≥ 0,30. "
      "Ambas com pós-processamento de 1 ha.\n\n" + q.to_markdown(index=False, disable_numparse=True) + "\n\n" + res
      + f"\n- **PAM**: {fm(t.area_ha_2025.sum())} ha\n\n*A PAM definiu a quantidade de rótulos (v10a/v10b): comparação otimista.*\n")
(ROOT / "reports/comparacao_unet_x_fusao.md").write_text(md, encoding="utf-8")
print(md)
print("UNET_SO_OK")
