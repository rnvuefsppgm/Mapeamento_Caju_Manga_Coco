"""22 (v10a) — Rótulos por RANKING com a área da PAM como prior (PU com prior de classe).

Nota = média (U-Net v10, XGBoost regional), 0-100. Por município:
  caju (1)     : pixels no topo do ranking até 0,6 x PAM, com nota >= 35, em manchas >= 1 ha
                 (+ polígonos de campo em Serra do Mel)
  não-caju (0) : pixels abaixo da posição de 3 x PAM (nota < t_neg, com t_neg <= 60)  OU  nota <= 10
  ignorado(255): faixa do meio, anel de 2 px em volta do caju, 100 m em volta dos polígonos de campo
A PAM só define QUANTOS rótulos entram; o mapa final não usa limiar por município.
Saída: data/rotulos/<rid>_rot.tif e reports/rotulos_rank.md
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

ROOT = Path(__file__).parent
V10 = ROOT.parent / "v10"
_s = importlib.util.spec_from_file_location("f10", V10 / "01_fontes.py"); f = importlib.util.module_from_spec(_s); _s.loader.exec_module(f)
OUT = ROOT / "data/rotulos"
FP, FN, MIN_POS, MAX_NEG = 0.6, 3.0, 35, 60


def nota_em_rank(sc, k):
    """Valor da nota na posição k (ordem decrescente); sc já sem nodata."""
    if k <= 0:
        return 101
    if k >= len(sc):
        return -1
    return int(np.partition(sc, len(sc) - k)[len(sc) - k])


def label(rid, pam_ha):
    with rasterio.open(V10 / f"predictions/2025/{rid}_prob.tif") as p:
        U, prof = p.read(1), p.profile
        tr, crs = p.transform, p.crs
    XR = rasterio.open(ROOT / f"data/fontes_a/{rid}_xr.tif").read(1)
    valid = (U != 255) & (XR != 255)
    S = np.where(valid, (U.astype("uint16") + XR) // 2, 255).astype("uint8")
    sc = S[valid]
    t_pos = max(nota_em_rank(sc, int(FP * pam_ha * 100)), MIN_POS)
    t_neg = min(nota_em_rank(sc, int(FN * pam_ha * 100)), MAX_NEG)
    pos = valid & (S >= t_pos)
    lab, n = ndimage.label(pos)
    if n:
        sz = np.bincount(lab.ravel()); keep = sz >= 100; keep[0] = False; pos = keep[lab]
    PL = np.zeros_like(pos)
    near = np.zeros_like(pos)
    if rid == "m2413359_SerradoMel":
        PL = f.polys(U.shape, tr, crs); near = f.polys(U.shape, tr, crs, buf=100.0) & ~PL
        pos = (pos & ~near) | PL
    neg = valid & ((S < t_neg) | (S <= 10)) & ~pos & ~near
    ring = ndimage.binary_dilation(pos, iterations=2) & ~pos
    out = np.full(U.shape, 255, "uint8"); out[neg & ~ring] = 0; out[pos] = 1
    prof.update(nodata=255, compress="deflate")
    with rasterio.open(OUT / f"{rid}_rot.tif", "w", **prof) as d:
        d.write(out, 1)
    return {"regiao": rid, "PAM_ha": pam_ha, "t_pos": t_pos, "t_neg": t_neg,
            "caju_ha": round(pos.sum() / 100, 1), "nao_ha": round((out == 0).sum() / 100, 1),
            "ign_ha": round(((out == 255) & valid).sum() / 100, 1)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pam = pd.read_csv(ROOT.parent / "dados_externos/ranking_caju_caatinga_2023_2025.csv").set_index("cod").area_ha_2025
    res = []
    for r in f.rids():
        res.append(label(r, float(pam.get(int(r[1:8]), 0)))); print(res[-1], flush=True)
    df = pd.DataFrame(res)
    (ROOT / "reports/rotulos_rank.md").write_text("# Rótulos por ranking (v10a)\n\n" + df.to_markdown(index=False), encoding="utf-8")
    print("ROTULOS_OK")


if __name__ == "__main__":
    main()
