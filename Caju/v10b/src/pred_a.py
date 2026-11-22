"""v10a-ft — predição: motor da v8/04b + TTA (original, espelho H, espelho V, rotação 180°) + membros mistos.
Pós-processamento: remove manchas < 1 ha e preenche buracos < 1 ha dentro do caju.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from scipy import ndimage

V8 = Path(__file__).resolve().parents[2] / "v8"
_s = importlib.util.spec_from_file_location("g04b", V8 / "04b_predict_emb_gpu.py")
g = importlib.util.module_from_spec(_s); _s.loader.exec_module(g)
from src.config import load_config  # noqa: E402
from src.models import build_model  # noqa: E402

CFG = load_config(str(V8.parent / "v7/config/config.yaml")); CFG["experiment"]["channels"] = "emb"
DEV = g.get_device()
MIN_PX = 100  # 1 ha


def load_members(files):
    mem = []
    for f in files:
        ck = torch.load(f, map_location="cpu", weights_only=False)
        m = build_model(CFG, in_channels=64); m.load_state_dict(ck["model_state"])
        mem.append((m.to(DEV).eval(), ck["stats"], float(ck["threshold"])))
    return mem


def tta_prob(raw, members):
    acc = 0
    for fl in ((), (1, 2)):  # TTA x2 (original + rotação 180°): 8 membros x 2 cabe no tempo
        r = np.ascontiguousarray(np.flip(raw, fl)) if fl else raw
        p = g.block_prob(r, members, CFG, DEV)
        acc = acc + (np.flip(p, tuple(a - 1 for a in fl)) if fl else p)
    return acc / 2


@torch.no_grad()
def predict(vrt, members, block=2048, margin=128):
    """Probabilidade 0-1 (float32, nan = sem dado) na grade do VRT."""
    with rasterio.open(vrt) as src:
        H, W = src.height, src.width
        prob = np.full((H, W), np.nan, "float32")
        for r0 in range(0, H, block):
            for c0 in range(0, W, block):
                rr, cc = max(0, r0 - margin), max(0, c0 - margin)
                win = Window(cc, rr, min(W, c0 + block + margin) - cc, min(H, r0 + block + margin) - rr)
                raw = src.read(window=win)
                if not (raw != 0).all(axis=0).any():
                    continue
                p = tta_prob(raw, members)
                ir, ic = r0 - rr, c0 - cc; hh, ww = min(block, H - r0), min(block, W - c0)
                prob[r0:r0 + hh, c0:c0 + ww] = p[ir:ir + hh, ic:ic + ww]
        return prob, src.profile


def pos_proc(cls):
    lab, n = ndimage.label(cls)
    if n:
        sz = np.bincount(lab.ravel()); keep = sz >= MIN_PX; keep[0] = False; cls = keep[lab]
    hol, n = ndimage.label(~cls)
    if n:
        sz = np.bincount(hol.ravel()); fill = sz < MIN_PX; fill[0] = False; cls = cls | fill[hol]
    return cls
