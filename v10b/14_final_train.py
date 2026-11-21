"""14 (v10a-ft) — Modelo FINAL por AJUSTE FINO: parte de v10/models/final_seed{S}.pt e treina com TODOS os chips
(campo + ranking v10a) por <epocas> (mediana da melhor época do CV v10a-ft), com aquecimento, encoder lento,
EMA, Lovász e sorteio 3x dos 5 fracos (src/ft.py). Salva os pesos EMA em models/final_seed{S}.pt.
Uso: python 14_final_train.py <epocas> <limiar>
"""
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import load_config, resolve_path
from src.dataset import CashewChipDataset
from src.metrics import get_loss
from src.models import build_model
from src.train_utils import get_device, set_seed
import src.ft as ft

V10M = Path(__file__).parent.parent / "v10/models"


def main():
    EP, THR = int(sys.argv[1]), float(sys.argv[2])
    cfg = load_config(None); dev = get_device()
    meta = pd.read_parquet(resolve_path(cfg, "metadata") / "chips_metadata_conc.parquet")
    ids = meta.chip_id.tolist()
    for i, seed in enumerate((42, 7, 123, 2024, 99)):  # v10b: 5 sementes
        set_seed(seed)
        os.environ["FT_INIT"] = str(V10M / f"final_seed{(42, 7, 123)[i % 3]}.pt")  # extras partem das finais da v10
        ck = ft.load_init(); st = ck["stats"]
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
                x, y = ft.gpu_prep(x, y, ds, True)  # v10b: preparo na GPU
                opt.zero_grad()
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = lf(m(x), y)
                sc.scale(loss).backward(); sc.step(opt); sc.update(); ema.update_parameters(m); tot += loss.item()
            print(f"seed {seed} ep {ep + 1}/{EP} loss {tot / len(dl):.4f}", flush=True)
        torch.save({"model_state": {k: v.cpu() for k, v in ema.module.state_dict().items()}, "stats": st,
                    "threshold": THR, "channels": "both", "epochs": EP, "seed": seed, "init": "v10 final (ajuste fino)"},
                   resolve_path(cfg, "models") / f"final_seed{seed}.pt")
    print("FINAL_OK")


if __name__ == "__main__":  # obrigatório no Windows (workers do DataLoader)
    main()
