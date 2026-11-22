"""30 — Treino U-Net do coco (receita caju v10b): CV espacial 5 folds  |  modelo final por sementes.

Modelo: smp.Unet(resnet18, 64 canais, 1 classe), iniciado com os pesos finais do caju v10b (mesma entrada).
Normalização: robusta por canal com as estatísticas do caju (mantém coerência com os pesos de partida).
Perda: focal + dice + 0,5·Lovász (ignore 255). AdamW lr 1e-4, wd 1e-3; aquecimento 5 ép. + cosseno até 1e-6;
encoder a 0,1× nas 10 primeiras épocas; EMA 0,998; AMP; flips/rot90 na GPU; chips com coco/ouro sorteados 2×.
Avaliação = labels_ouro (verificado) + não-coco automático de evidência forte (QC 1,00); coco só do ouro.

Uso:  python 30_treino.py cv [max_epocas=60] [paciencia=15] [folds=0,1,2,3,4]
      python 30_treino.py final [epocas=100] [sementes=42,7,123,2024,99]
Saídas: v1/models/cv/fold{k}.pt, v1/reports/oof_fold{k}.npz, v1/models/final_seed{s}.pt, logs/30_*.log
"""
import json
import math
import sys
import time

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from common import cfg, ver

c = cfg()
T = c["treino"]
CACHE = ver(c, "data", "cache", "x").parent
CAJU = c["externos"]["caju_models"]
WARM, ENC_SLOW, ENC_F, EMA_D, LOV_W, LR, WD, BS = 5, 10, 0.1, 0.998, 0.5, 1e-4, 1e-3, 16
dev = "cuda"


class Chips(Dataset):
    def __init__(self, idx, avaliar=False):
        self.idx, self.avaliar = np.asarray(idx), avaliar
        self.F = self.LB = self.LO = None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        if self.F is None:  # abre o memmap no worker
            self.F = np.load(CACHE / "features.npy", mmap_mode="r")
            self.LB = np.load(CACHE / "labels.npy", mmap_mode="r")
            self.LO = np.load(CACHE / "labels_ouro.npy", mmap_mode="r")
        k = self.idx[i]
        y = np.array(self.LB[k])
        if self.avaliar:   # canal 0 = ouro sobre automático (seleção); canal 1 = ouro + não-coco automático (coco só ouro)
            lo = np.array(self.LO[k])
            y = np.stack([np.where(lo != 255, lo, y), np.where(lo != 255, lo, np.where(y == 0, 0, 255))]).astype(np.uint8)
        return torch.from_numpy(np.array(self.F[k])), torch.from_numpy(y)


def norm_stats(seed=42):
    st = torch.load(f"{CAJU}/final_seed{seed}.pt", map_location="cpu", weights_only=False)["stats"]
    lo = torch.tensor([s["lo"] for s in st["per_channel"]], device=dev)[None, :, None, None]
    hi = torch.tensor([s["hi"] for s in st["per_channel"]], device=dev)[None, :, None, None]
    return st, lo, hi


def prep(x, y, lo, hi, aug):
    x = (((x.to(dev, non_blocking=True).float() - 128) / 127 - lo) / (hi - lo)).clamp_(0, 1)
    y = y.to(dev, non_blocking=True).long()
    x = torch.where((x == 0).all(1, keepdim=True), torch.zeros_like(x), x)
    if aug:
        if torch.rand(1) < 0.5:
            x, y = x.flip(-1), y.flip(-1)
        if torch.rand(1) < 0.5:
            x, y = x.flip(-2), y.flip(-2)
        k = int(torch.randint(0, 4, (1,)))
        x, y = torch.rot90(x, k, (-2, -1)).contiguous(), torch.rot90(y, k, (-2, -1)).contiguous()
    return x, y


def modelo(seed):
    m = smp.Unet("resnet18", encoder_weights=None, in_channels=64, classes=1)
    m.load_state_dict(torch.load(f"{CAJU}/final_seed{seed}.pt", map_location="cpu", weights_only=False)["model_state"])
    return m.to(dev)


focal = smp.losses.FocalLoss("binary", ignore_index=255)
dice = smp.losses.DiceLoss("binary", ignore_index=255, from_logits=True)
lov = smp.losses.LovaszLoss("binary", ignore_index=255, from_logits=True)
perda = lambda p, y: focal(p, y) + dice(p, y) + LOV_W * lov(p, y)


def treinar_epoca(m, ema, opt, sc, dl, lo, hi, ep, eps):
    k = (ep + 1) / WARM if ep < WARM else 1e-6 / LR + (1 - 1e-6 / LR) * 0.5 * (1 + math.cos(math.pi * (ep - WARM) / max(1, eps - WARM)))
    for g in opt.param_groups:
        g["lr"] = LR * k * (ENC_F if g["enc"] and ep < ENC_SLOW else 1)
    m.train(); tot = 0
    for x, y in dl:
        x, y = prep(x, y, lo, hi, True)
        y = y.unsqueeze(1)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = perda(m(x), y)
        opt.zero_grad(set_to_none=True); sc.scale(loss).backward(); sc.step(opt); sc.update()
        ema.update_parameters(m); tot += loss.item()
    return tot / len(dl)


@torch.no_grad()
def avaliar(net, dl, lo, hi, coletar=False):
    net.eval(); M = np.zeros((2, 3)); P, Y, YO = [], [], []
    rng = np.random.default_rng(0)
    for x, y in dl:
        x = (((x.to(dev).float() - 128) / 127 - lo) / (hi - lo)).clamp_(0, 1)
        y = y.to(dev).long()
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(net(x)).float().squeeze(1)
        for j in range(2):
            yj = y[:, j]; v = yj != 255
            pr, yt = p[v] >= 0.5, yj[v] == 1
            M[j] += [int((pr & yt).sum()), int((pr & ~yt).sum()), int((~pr & yt).sum())]
        if coletar:
            v = y[:, 0] != 255
            pv, yv, yo = p[v].cpu().numpy(), y[:, 0][v].cpu().numpy().astype(np.uint8), y[:, 1][v].cpu().numpy().astype(np.uint8)
            keep = (yv == 1) | (yo == 1) | (rng.random(yv.size) < 0.05)       # todo coco + 5 % do resto
            P.append(pv[keep].astype(np.float16)); Y.append(yv[keep]); YO.append(yo[keep])
    iou = [m[0] / max(1, m.sum()) for m in M]   # [comb, ouro]
    cat = lambda L: np.concatenate(L) if L else None
    return iou, cat(P), (cat(Y), cat(YO))


def otimizador(m):
    enc = [p for n, p in m.named_parameters() if n.startswith("encoder.")]
    res = [p for n, p in m.named_parameters() if not n.startswith("encoder.")]
    return torch.optim.AdamW([{"params": enc, "enc": True}, {"params": res, "enc": False}], lr=LR, weight_decay=WD)


def loader(idx, treino, meta=None, aval=False):
    ds = Chips(idx, avaliar=aval)
    if treino:
        w = np.where((meta.coco_fraction.values[idx] > 0) | (meta.source.values[idx] == "field"), 2.0, 1.0)
        return DataLoader(ds, BS, sampler=WeightedRandomSampler(w, len(idx), replacement=True), num_workers=2,
                          pin_memory=True, persistent_workers=True, drop_last=True)
    return DataLoader(ds, BS, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)


def folds(meta, K):
    """Blocos de 10 km → folds; blocos com ouro coco distribuídos em rodízio por macro-região."""
    rng = np.random.default_rng(42)
    b = meta.groupby("block_id").agg(macro=("macro", "first"), oc=("ouro_coco_px", "sum")).reset_index()
    f = {}
    for mac, g in b.groupby("macro"):
        com = g[g.oc > 0].sort_values("oc", ascending=False)
        for i, bid in enumerate(com.block_id):
            f[bid] = i % K
        for bid in g[g.oc == 0].block_id:
            f[bid] = int(rng.integers(K))
    return meta.block_id.map(f).values


def main():
    modo = sys.argv[1]
    meta = pd.read_parquet(CACHE / "chips_metadata.parquet")
    LO = np.load(CACHE / "labels_ouro.npy", mmap_mode="r")
    meta["ouro_coco_px"] = [int((LO[k] == 1).sum()) for k in meta.cache_idx]
    st, lo, hi = norm_stats()
    rep = ver(c, "reports", "x").parent; mod = ver(c, "models", "cv", "x").parent; mod.mkdir(parents=True, exist_ok=True)
    if modo == "cv":
        E = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        PAC = int(sys.argv[3]) if len(sys.argv) > 3 else 15
        FS = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else list(range(T["cv_folds"]))
        meta["fold"] = folds(meta, T["cv_folds"])
        meta[["chip_id", "block_id", "macro", "fold"]].to_csv(rep / "cv_folds.csv", index=False)
        for k in FS:
            if (rep / f"oof_fold{k}.npz").exists():   # retomável: pula fold já concluído
                print(f"FOLD {k}: já concluído, pulando", flush=True); continue
            tr, va = np.flatnonzero(meta.fold != k), np.flatnonzero(meta.fold == k)
            torch.manual_seed(42); m = modelo(42); opt = otimizador(m); sc = torch.amp.GradScaler()
            ema = torch.optim.swa_utils.AveragedModel(m, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(EMA_D), use_buffers=True)
            dtr, dva = loader(tr, True, meta), loader(va, False, aval=True)
            best, bep, t0 = -1, -1, time.time()
            for ep in range(E):
                l = treinar_epoca(m, ema, opt, sc, dtr, lo, hi, ep, E)
                (iou, iou_o), _, _ = avaliar(ema.module, dva, lo, hi)
                if iou > best:
                    best, bep = iou, ep
                    torch.save({"model_state": {a: b.cpu() for a, b in ema.module.state_dict().items()}, "stats": st,
                                "epoch": ep, "iou_val": iou}, mod / f"fold{k}.pt")
                print(f"fold {k} ep {ep:02d} loss {l:.4f} IoU_val@0.5 {iou:.3f} (só ouro {iou_o:.3f}) (melhor {best:.3f}@{bep}) {time.time() - t0:.0f}s", flush=True)
                if ep - bep >= PAC and ep >= WARM + ENC_SLOW:
                    break
            m.load_state_dict(torch.load(mod / f"fold{k}.pt", weights_only=False)["model_state"])
            _, P, (Y, YO) = avaliar(m, dva, lo, hi, coletar=True)
            np.savez_compressed(rep / f"oof_fold{k}.npz", p=P, y=Y, y_ouro=YO, macro=np.array(meta.macro.values[va][0]), best_epoch=bep, iou=best)
            print(f"FOLD {k}: melhor IoU@0.5 {best:.3f} na época {bep}; val {len(va)} chips", flush=True)
    else:
        E = int(sys.argv[2]) if len(sys.argv) > 2 else T["epocas_final"]
        SE = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else T["sementes"]
        thr = json.loads((rep / "cv_resumo.json").read_text())["limiar"] if (rep / "cv_resumo.json").exists() else 0.5
        idx = np.arange(len(meta))
        for s in SE:
            torch.manual_seed(s); m = modelo(s); opt = otimizador(m); sc = torch.amp.GradScaler()
            ema = torch.optim.swa_utils.AveragedModel(m, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(EMA_D), use_buffers=True)
            dtr, t0 = loader(idx, True, meta), time.time()
            for ep in range(E):
                l = treinar_epoca(m, ema, opt, sc, dtr, lo, hi, ep, E)
                if ep % 10 == 9 or ep == E - 1:
                    print(f"seed {s} ep {ep:03d} loss {l:.4f} {time.time() - t0:.0f}s", flush=True)
            torch.save({"model_state": {a: b.cpu() for a, b in ema.module.state_dict().items()}, "stats": st,
                        "threshold": thr, "epochs": E, "seed": s, "init": f"caju_v10b_final_seed{s}"},
                       ver(c, "models", f"final_seed{s}.pt"))
            print(f"FINAL seed {s} salvo", flush=True)


if __name__ == "__main__":
    main()
