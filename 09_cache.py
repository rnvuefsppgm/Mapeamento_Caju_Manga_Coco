"""09 — Cache memmap único em 8 bits (formato do caju v10b: dados/cache/features_u8.npy + labels.npy) a partir
dos chips do 08, convertendo CÓDIGOS -> rótulo {0,1,255} pela variante de negativos:
  denso : 1 -> 1 ; 3,4 -> 0 ; resto -> 255   (área amostrada tida como mapeada por completo; caju v7)
  mb    : 1 -> 1 ; 4 -> 0   ; resto -> 255   (só negativos confiáveis MapBiomas; caju v10b)
uso: python 09_cache.py [denso|mb]   (padrão: denso). Grava metadata/chips_metadata_all.parquet com cache_idx.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path, setup_logging

cfg = load_config(Path(__file__).parent / "config/config.yaml")
log = setup_logging("09_cache", cfg)
NEG = {"denso": [3, 4], "mb": [4]}


def main(var):
    meta = pd.read_parquet(resolve_path(cfg, "metadata") / "chips_metadata_campo.parquet").reset_index(drop=True)
    c, h, w = np.load(meta.loc[0, "path"])["features"].shape
    out = resolve_path(cfg, "cache")
    F = np.lib.format.open_memmap(out / "features_u8.npy", mode="w+", dtype="uint8", shape=(len(meta), c, h, w))
    L = np.lib.format.open_memmap(out / "labels.npy", mode="w+", dtype="uint8", shape=(len(meta), h, w))
    fr = []
    for i, p in enumerate(meta.path):
        d = np.load(p)
        k = d["codes"]
        lab = np.full(k.shape, 255, "uint8")
        lab[np.isin(k, NEG[var])] = 0
        lab[k == 1] = 1
        lab[d["valid"] == 0] = 255
        F[i], L[i] = d["features"], lab
        fr.append(((lab == 1).mean(), (lab == 0).mean()))
    F.flush(); L.flush()
    meta["cache_idx"] = np.arange(len(meta))
    meta["cashew_fraction"], meta["background_fraction"] = zip(*fr)  # nomes do caju (src/dataset.py)
    meta["neg_variant"] = var
    meta.to_parquet(resolve_path(cfg, "metadata") / "chips_metadata_all.parquet", index=False)
    pos, neg = meta.cashew_fraction.mean(), meta.background_fraction.mean()
    log.info(f"RESUMO cache ({var}): {len(meta)} chips, {c} canais, {len(meta) * c * h * w / 1e9:.2f} GB; "
             f"pixels manga {100 * pos:.1f}% · não-manga {100 * neg:.1f}% · ignorado {100 * (1 - pos - neg):.1f}%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "denso")
