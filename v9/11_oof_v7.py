"""11 (v9) — Candidato C1: predições fora do fold do `unet_emb` da v7 avaliadas nos chips de campo da v9.

Usa os 5 modelos salvos (../v7/models/ensemble_unet_emb/foldN.pt), cada um nos blocos de teste do seu fold
(`test_blocks` no checkpoint). Rótulos = chips de campo da v9 (com faixa de exclusão), para comparar
C1–C4 com a mesma referência. Saída: reports/oof/unet_emb_v7_foldN.npz (p, t).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import load_config, resolve_path
from src.dataset import apply_norm, load_chip
from src.models import build_model
from src.train_utils import get_device

ROOT = Path(__file__).parent
cfg = load_config(None)
dev = get_device()
meta = pd.read_parquet(resolve_path(cfg, "metadata") / "chips_metadata_all.parquet")
meta = meta[meta.variant == "field"]
od = ROOT / "reports" / "oof"
od.mkdir(parents=True, exist_ok=True)
for f in sorted((ROOT.parent / "v7" / "models" / "ensemble_unet_emb").glob("fold*.pt")):
    ck = torch.load(f, map_location="cpu", weights_only=False)
    m = build_model(cfg, in_channels=len(ck["stats"]["per_channel"]))
    m.load_state_dict(ck["model_state"])
    m.to(dev).eval()
    te = meta[meta.block_id.isin(ck["test_blocks"])]
    P, T = [], []
    with torch.no_grad():
        for _, r in te.iterrows():
            x, y = load_chip(r, cfg)
            x = torch.from_numpy(apply_norm(x, ck["stats"])[None]).float().to(dev)
            P.append(torch.sigmoid(m(x))[0, 0].cpu().numpy().astype("float16"))
            T.append(y.astype("uint8"))
    np.savez_compressed(od / f"unet_emb_v7_{f.stem}.npz", p=np.stack(P), t=np.stack(T))
    print(f.stem, len(te), "chips de teste", flush=True)
