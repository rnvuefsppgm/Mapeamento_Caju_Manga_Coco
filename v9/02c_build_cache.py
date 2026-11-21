"""02c (v7) — Junta todos os chips .npz num único cache memmap (leitura rápida no treino).

Motivo: abrir 1.033 arquivos .npz (zip) por época deixava a GPU ociosa (CPU/IO).
Saída: data/cache/features.npy (N,C,H,W float16), labels.npy (N,H,W uint8) e a
coluna ``cache_idx`` no chips_metadata.parquet. O dataset lê só os canais usados.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import load_config, resolve_path, setup_logging


def main() -> None:
    cfg = load_config(None)
    log = setup_logging("02c_build_cache", cfg)
    mpath = resolve_path(cfg, "metadata") / "chips_metadata_all.parquet"  # v9
    meta = pd.read_parquet(mpath).reset_index(drop=True)
    first = np.load(meta.loc[0, "path"])
    c, h, w = first["features"].shape
    out = resolve_path(cfg, "chips").parent / "cache"
    out.mkdir(parents=True, exist_ok=True)
    feats = np.lib.format.open_memmap(out / "features.npy", mode="w+", dtype="float16",
                                      shape=(len(meta), c, h, w))
    labels = np.lib.format.open_memmap(out / "labels.npy", mode="w+", dtype="uint8",
                                       shape=(len(meta), h, w))
    for i, p in enumerate(tqdm(meta["path"], desc="cache", unit="chip")):
        d = np.load(p)
        feats[i] = d["features"]
        labels[i] = d["label"]
    feats.flush(); labels.flush()
    del feats, labels
    meta["cache_idx"] = np.arange(len(meta))
    meta.to_parquet(mpath, index=False)
    for v in ("A", "B"):  # v9: campo + pseudo variante v (mesmo cache)
        meta[meta["variant"].isin(["field", v])].to_parquet(
            mpath.with_name(f"chips_metadata_{v}.parquet"), index=False)
    log.info("Cache: %d chips, %d canais -> %s (%.1f GB)", len(meta), c, out,
             len(meta) * c * h * w * 2 / 1e9)


if __name__ == "__main__":
    main()
