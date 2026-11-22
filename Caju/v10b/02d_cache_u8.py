"""02d (v10b) — Converte o cache de treino (features.npy, float16) para 8 bits (features_u8.npy): u = round(x*127)+128.
Exato (os embeddings já são quantizados em passos de 1/127) e com metade do tamanho, para caber na RAM livre.
O src/dataset.py usa o features_u8.npy automaticamente quando ele existe (e o GPU_AUG=1 exige ele).
"""
from pathlib import Path

import numpy as np

from src.config import load_config, resolve_path

cfg = load_config(None)
d = resolve_path(cfg, "chips").parent / "cache"
F = np.load(d / "features.npy", mmap_mode="r")
U = np.lib.format.open_memmap(d / "features_u8.npy", mode="w+", dtype="uint8", shape=F.shape)
for i in range(0, len(F), 64):
    U[i:i + 64] = np.clip(np.round(np.asarray(F[i:i + 64], dtype=np.float32) * 127) + 128, 1, 255).astype("uint8")
U.flush()
print("CACHE_U8_OK", U.shape)
