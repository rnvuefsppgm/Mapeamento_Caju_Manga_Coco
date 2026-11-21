"""03 (v10) — Amostras: chips de campo (Serra do Mel, com faixa de exclusão) + chips dos 25 municípios
a partir dos rótulos por concordância (até 20 com caju + 20 sem caju por município; >= 50% rotulado).
Saída: metadata/chips_metadata_all.parquet (variant: field | conc); depois 02c gera o cache.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from src.chip_generator import _spatial_block_id
from src.config import ensure_dirs, load_config, resolve_path, setup_logging

ROOT = Path(__file__).parent
_s = importlib.util.spec_from_file_location("c9", ROOT / "02_chips_v9.py"); c9 = importlib.util.module_from_spec(_s); _s.loader.exec_module(c9)
EMB = ROOT.parent / "v8/data/ee_exports/2025/embeddings"
PER = 20


def main():
    cfg = load_config(None); ensure_dirs(cfg); log = setup_logging("03_chips", cfg)
    field = c9.field_chips(cfg, log)
    size, out = cfg["chips"]["size"], resolve_path(cfg, "chips")
    rng, rows = np.random.default_rng(7), []
    for lf in sorted((ROOT / "data/rotulos").glob("m*_rot.tif")):
        rid = lf.name.replace("_rot.tif", "")
        with rasterio.open(lf) as d:
            L = d.read(1)
        cp, cn = [], []
        for r0 in range(0, L.shape[0] - size + 1, size):
            for c0 in range(0, L.shape[1] - size + 1, size):
                w = L[r0:r0 + size, c0:c0 + size]
                if (w != 255).mean() < 0.5:
                    continue
                (cp if (w == 1).mean() >= 0.02 else cn).append((r0, c0))
        npos = min(PER, len(cp))  # v10b: negativos <= positivos + 5 (evita reforçar a subestimação)
        pick = [cp[i] for i in rng.permutation(len(cp))[:npos]] + [cn[i] for i in rng.permutation(len(cn))[:npos + 5]]
        with rasterio.open(EMB / f"{rid}.vrt") as e:
            for k, (r0, c0) in enumerate(pick):
                fe = c9.dec_v8(e.read(window=Window(c0, r0, size, size)))
                valid = np.isfinite(fe).all(axis=0)
                lab = L[r0:r0 + size, c0:c0 + size].copy(); lab[~valid] = 255
                cid = f"conc_{rid}_{k:03d}"; path = out / f"{cid}.npz"
                np.savez(path, features=np.nan_to_num(fe, nan=0.0).astype("float16"), label=lab, valid=valid.astype("uint8"))
                rows.append({"chip_id": cid, "path": str(path), "municipality": rid, "valid_fraction": float(valid.mean()),
                             "cashew_fraction": float((lab == 1).mean()), "background_fraction": float((lab == 0).mean()),
                             "n_channels": 64, "chip_size": size, "resolution": 10, "crs": cfg["project"]["crs_working"],
                             "block_id": _spatial_block_id(e.transform, r0, c0, size, cfg["split"]["block_size_km"] * 1000.0),
                             "is_negative": bool((lab == 1).sum() == 0), "year": 2025, "source": "pseudo", "variant": "conc"})
        log.info("%s: %d chips (cand. caju=%d, não=%d)", rid, len(pick), len(cp), len(cn))
    df = pd.concat([field, pd.DataFrame(rows)], ignore_index=True)
    df.to_parquet(resolve_path(cfg, "metadata") / "chips_metadata_all.parquet", index=False)
    log.info("Total: %s", df.groupby("variant").size().to_dict())


if __name__ == "__main__":
    main()
