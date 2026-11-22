"""15 (v1) — Modelo FINAL por ajuste fino (receita caju v10b/14_final_train.py): todos os chips v1, épocas fixas,
5 sementes, aquecimento 5, encoder lento 10, EMA, + Lovász, preparo na GPU. Parte do init vencedor do CV A/B.
Uso: python 15_final.py <init.pt> <limiar>   -> modelos/final_seed{S}.pt
"""
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import load_config, resolve_path
from src.dataset import CashewChipDataset, compute_norm_stats
from src.metrics import get_loss
from src.models import build_model
from src.train_utils import get_device, set_seed
import src.ft as ft


def main():
    INIT, THR = sys.argv[1], float(sys.argv[2])
    cfg = load_config(None); dev = get_device(); F = cfg["treino_v1"]["final"]; EP = F["epocas"]
    meta = pd.read_parquet(resolve_path(cfg, "metadata") / "chips_metadata_v1.parquet")
    ids = meta.chip_id.tolist()
    st = compute_norm_stats(meta, ids, cfg)
    os.environ["FT_INIT"] = INIT
    for seed in F["sementes"]:
        set_seed(seed)
        ck = ft.load_init()
        ds = CashewChipDataset(meta, ids, st, cfg, augment=True)
        dl = DataLoader(ds, batch_size=cfg["training"]["batch_size"], sampler=ft.sampler(meta, ids), drop_last=True,
                        num_workers=cfg["training"]["num_workers"], pin_memory=True, persistent_workers=True)
        m = build_model(cfg, in_channels=64); m.load_state_dict(ck["model_state"]); m = m.to(dev)
        opt = ft.optimizer(m, cfg); ema = ft.ema(m)
        lf = ft.with_lovasz(get_loss(cfg["training"]["loss"], cfg["chips"]["label_ignore_value"]),
                            cfg["chips"]["label_ignore_value"])
        sc = torch.amp.GradScaler()
        for ep in range(EP):
            ft.set_lr(opt, ep, EP, cfg); m.train(); tot = 0.0
            for x, y in dl:
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                x, y = ft.gpu_prep(x, y, ds, True)
                opt.zero_grad()
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = lf(m(x), y)
                sc.scale(loss).backward(); sc.step(opt); sc.update(); ema.update_parameters(m); tot += loss.item()
            if (ep + 1) % 10 == 0:
                print(f"seed {seed} ep {ep + 1}/{EP} loss {tot / len(dl):.4f}", flush=True)
        torch.save({"model_state": {k: v.cpu() for k, v in ema.module.state_dict().items()}, "stats": st,
                    "threshold": THR, "channels": "both", "epochs": EP, "seed": seed, "init": INIT},
                   resolve_path(cfg, "models") / f"final_seed{seed}.pt")
    print("FINAL_OK")


if __name__ == "__main__":  # obrigatório no Windows (workers do DataLoader)
    main()
