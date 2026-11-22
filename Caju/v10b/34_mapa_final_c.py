"""34 (v10b) — MAPA FINAL, opção (c), liberada pelo usuário em 21/11/2026: fusão U-Net+XGB regional (0,6/0,4, limiar
0,30) só no sertão CE/RN; só U-Net (limiar 0,40) no litoral CE e no PI/PE. Pós-processamento de 1 ha nos dois.
Saídas: predictions/2025/<rid>_class_final.tif; predictions/caju_2025_25mun_FINAL_classe.tif e
caju_2025_25mun_FINAL_prob.tif (EPSG:32724, 10 m, COG; prob = fusão no sertão, U-Net no resto);
reports/area_caju_2025_final.csv
"""
import importlib.util
import shutil
from pathlib import Path

import pandas as pd
import rasterio
from rasterio.merge import merge

ROOT = Path(__file__).parent
_s = importlib.util.spec_from_file_location("f10", ROOT.parent / "v10/01_fontes.py"); f = importlib.util.module_from_spec(_s); _s.loader.exec_module(f)
OD = ROOT / "predictions/2025"
FUSAO = {"sertao_CE_RN"}
rows, cls_f, prob_f = [], [], []
for rid in f.rids():
    reg = f.REG_OF[int(rid[1:8])]
    fus = reg in FUSAO
    src_c = OD / (f"{rid}_class.tif" if fus else f"{rid}_class_unet.tif")
    src_p = OD / (f"{rid}_final.tif" if fus else f"{rid}_prob.tif")
    shutil.copyfile(src_c, OD / f"{rid}_class_final.tif")
    with rasterio.open(src_c) as d:
        a = d.read(1)
    rows.append({"regiao": rid, "reg": reg, "metodo": "fusão 0,6/0,4 ≥ 0,30" if fus else "só U-Net ≥ 0,40",
                 "caju_ha": round(float((a == 1).sum()) / 100, 1)})
    cls_f.append(OD / f"{rid}_class_final.tif"); prob_f.append(src_p)
for fs, nome in ((cls_f, "classe"), (prob_f, "prob")):
    srcs = [rasterio.open(x) for x in fs]
    arr, tr = merge(srcs, nodata=255, method="first")
    pr = dict(driver="COG", height=arr.shape[1], width=arr.shape[2], count=1, dtype="uint8", crs=srcs[0].crs,
              transform=tr, nodata=255, compress="DEFLATE", predictor=2, overviews="AUTO", BIGTIFF="IF_SAFER")
    with rasterio.open(ROOT / f"predictions/caju_2025_25mun_FINAL_{nome}.tif", "w", **pr) as d:
        d.write(arr[0], 1)
    for s in srcs:
        s.close()
df = pd.DataFrame(rows)
p = pd.read_csv(ROOT.parent / "dados_externos/ranking_caju_caatinga_2023_2025.csv")[["cod", "nome", "area_ha_2025"]]
df["cod"] = df.regiao.str[1:8].astype(int); df = df.merge(p, on="cod"); df["razao_PAM"] = (df.caju_ha / df.area_ha_2025).round(2)
df.to_csv(ROOT / "reports/area_caju_2025_final.csv", index=False, encoding="utf-8")
print(f"total {df.caju_ha.sum():,.0f} ha | corr {df[['caju_ha', 'area_ha_2025']].corr().iloc[0, 1]:.2f}")
print("FINAL_C_OK")
