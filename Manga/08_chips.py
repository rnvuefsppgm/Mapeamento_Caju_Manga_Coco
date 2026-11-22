"""08 — Chips de campo (256 px, stride 96) dos embeddings 2025 crus em 8 bits + raster de CÓDIGOS (07).

Diferença do caju: guarda o chip em uint8 (valor cru x*127+128; NoData -> 128 = 0,0 decodificado, igual ao
nan_to_num do caju) e os códigos, não o rótulo final; o 09 converte códigos -> {0,1,255} pela variante escolhida.
Saída: dados/chips/campo_<rid>_<k>.npz (features uint8 64xHxW, codes uint8 HxW) + metadata/chips_metadata_campo.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from src.chip_generator import _spatial_block_id
from src.config import load_config, resolve_path, setup_logging

cfg = load_config(Path(__file__).parent / "config/config.yaml")
log = setup_logging("08_chips", cfg)
C = cfg["manga"]["chips_campo"]
SIZE = cfg["chips"]["size"]


def main():
    out, rows = resolve_path(cfg, "chips"), []
    for lf in sorted(resolve_path(cfg, "rotulos").glob("*_campo.tif")):
        rid = lf.name.replace("_campo.tif", "")
        with rasterio.open(lf) as d:
            K = d.read(1)
        ys, xs = np.where(K == 1)
        r_lo, r_hi = max(0, ys.min() - SIZE), min(K.shape[0] - SIZE, ys.max())
        c_lo, c_hi = max(0, xs.min() - SIZE), min(K.shape[1] - SIZE, xs.max())
        with rasterio.open(resolve_path(cfg, "embeddings") / f"{rid}.vrt") as e:
            k = 0
            for r0 in range(r_lo, r_hi + 1, C["stride"]):
                for c0 in range(c_lo, c_hi + 1, C["stride"]):
                    w = K[r0:r0 + SIZE, c0:c0 + SIZE]
                    if np.isin(w, [1, 3, 4]).mean() < C["min_rotulado"] or (w == 255).mean() > 0.3:
                        continue
                    f = e.read(window=Window(c0, r0, SIZE, SIZE))
                    valid = (f != 0).all(axis=0)
                    f[:, ~valid] = 128
                    cid = f"campo_{rid}_{k:05d}"
                    path = out / f"{cid}.npz"
                    np.savez(path, features=f, codes=w, valid=valid.astype("uint8"))
                    rows.append({"chip_id": cid, "path": str(path), "municipality": rid, "row": r0, "col": c0,
                                 "valid_fraction": float(valid.mean()), "manga_fraction": float((w == 1).mean()),
                                 "neg_denso_fraction": float((w == 3).mean()), "neg_mb_fraction": float((w == 4).mean()),
                                 "n_channels": 64, "chip_size": SIZE, "resolution": 10, "crs": cfg["manga"]["crs"],
                                 "block_id": _spatial_block_id(e.transform, r0, c0, SIZE, C["bloco_cv_km"] * 1000.0),
                                 "is_negative": bool((w == 1).sum() == 0), "year": 2025, "source": "field",
                                 "variant": "field"})
                    k += 1
        log.info(f"{rid}: {k} chips")
    df = pd.DataFrame(rows)
    df.to_parquet(resolve_path(cfg, "metadata") / "chips_metadata_campo.parquet", index=False)
    log.info(f"RESUMO: {len(df)} chips ({int((~df.is_negative).sum())} com manga), {df.block_id.nunique()} blocos de "
             f"{C['bloco_cv_km']} km; chips por bloco: {df.block_id.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
