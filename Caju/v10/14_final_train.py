"""14 (v10) — Modelo FINAL: U-Net só embeddings, TODOS os chips (campo + concordância), épocas fixas
(mediana da melhor época do CV), 3 sementes. Limiar = global fora do fold (reports/oof_resumo.md).
Saída: models/final_seed{S}.pt (pesos + normalização + limiar). Não faz predição (parada combinada).
Uso: python 14_final_train.py <epocas> <limiar>
"""
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import load_config, resolve_path
from src.dataset import CashewChipDataset, compute_norm_stats
from src.metrics import get_loss
from src.models import build_model
from src.train_utils import get_device, set_seed

def main():
    EP, THR = int(sys.argv[1]), float(sys.argv[2])
    cfg = load_config(None); dev = get_device()
    meta = pd.read_parquet(resolve_path(cfg, "metadata") / "chips_metadata_conc.parquet")
    ids = meta.chip_id.tolist()
    for seed in (42, 7, 123):
        set_seed(seed)
        st = compute_norm_stats(meta, ids, cfg)
        ds = CashewChipDataset(meta, ids, st, cfg, augment=True)
        dl = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True, drop_last=True,
                        num_workers=cfg["training"]["num_workers"], pin_memory=True, persistent_workers=True)
        m = build_model(cfg, in_channels=64).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EP)
        lf = get_loss(cfg["training"]["loss"], cfg["chips"]["label_ignore_value"]); sc = torch.amp.GradScaler()
        for ep in range(EP):
            m.train(); tot = 0.0
            for x, y in dl:
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                opt.zero_grad()
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = lf(m(x), y)
                sc.scale(loss).backward(); sc.step(opt); sc.update(); tot += loss.item()
            sch.step()
            print(f"seed {seed} ep {ep + 1}/{EP} loss {tot / len(dl):.4f}", flush=True)
        torch.save({"model_state": {k: v.cpu() for k, v in m.state_dict().items()}, "stats": st, "threshold": THR,
                    "channels": "both", "epochs": EP, "seed": seed}, resolve_path(cfg, "models") / f"final_seed{seed}.pt")
    print("FINAL_OK")


if __name__ == "__main__":  # obrigatório no Windows (workers do DataLoader)
    main()
