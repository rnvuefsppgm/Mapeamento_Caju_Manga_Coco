"""40 — Mapa de coco 2025 por município: ensemble dos modelos finais + TTA ×2, na GPU, em blocos com margem.

Membros: v1/models/final_seed*.pt (ou os indicados). Blocos de 1024 px + margem de 64 px (sem costura).
Pós-processamento: classe = prob ≥ limiar (cv_resumo.json); zera classes MapBiomas seguras de não-coco;
remove manchas < 1 ha; fora do município = 255. Retomável (pula município pronto).
Saídas: v1/predictions/2025/<regiao>_prob.tif (0–100) e _classe.tif (0/1; 255 fora)
Uso: python 40_predicao.py <lista> [membros.pt separados por vírgula]
"""
import json
import sys

import geopandas as gpd
import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
from rasterio.features import rasterize
from rasterio.windows import Window
from scipy import ndimage

from common import cfg, ver

c = cfg()
lista = sys.argv[1]
dev = "cuda"
B, M = 1024, 64
rep = ver(c, "reports", "x").parent
mods = sys.argv[2].split(",") if len(sys.argv) > 2 else sorted(str(p) for p in ver(c, "models", "x").parent.glob("final_seed*.pt"))
thr = json.loads((rep / "cv_resumo.json").read_text())["limiar"] if (rep / "cv_resumo.json").exists() else 0.5
out = ver(c, "predictions", "teste" if lista == "teste" else "2025", "x").parent
out.mkdir(parents=True, exist_ok=True)
emb = ver(c, "data", "embeddings", "2025", "x").parent
fon = ver(c, "data", "fontes", "x").parent
regioes = [l.split() for l in ver(c, "data", "aoi", f"regions_{lista}.txt").read_text().split("\n") if l.strip()]
mun = gpd.read_file(c["externos"]["municipios_shp"])
colc = next(k for k in mun.columns if k.upper().startswith("CD_MUN"))

nets = []
for f in mods:
    ck = torch.load(f, map_location="cpu", weights_only=False)
    n = smp.Unet("resnet18", encoder_weights=None, in_channels=64, classes=1)
    n.load_state_dict(ck["model_state"]); nets.append(n.to(dev).eval().half())
pc = ck["stats"]["per_channel"]
lo = torch.tensor([s["lo"] for s in pc], device=dev)[:, None, None].half()
hi = torch.tensor([s["hi"] for s in pc], device=dev)[:, None, None].half()
print(f"{len(nets)} membros, limiar {thr}", flush=True)


@torch.no_grad()
def prever(a):
    x = torch.from_numpy(a).to(dev)
    v = x[0] != 0
    x = (((x.half() - 128) / 127 - lo) / (hi - lo)).clamp_(0, 1)[None]
    h, w = x.shape[-2:]
    ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
    x = torch.nn.functional.pad(x, (0, pw, 0, ph))
    p = 0
    for n in nets:
        p = p + torch.sigmoid(n(x)) + torch.sigmoid(n(x.flip(-1))).flip(-1)
    p = (p / (2 * len(nets)))[0, 0, :h, :w]
    return (p * 100).round().byte().where(v, torch.zeros_like(v, dtype=torch.uint8)).cpu().numpy(), v.cpu().numpy()


for rid, crs in regioes:
    fp, fc = out / f"{rid}_prob.tif", out / f"{rid}_classe.tif"
    if fc.exists():
        continue
    with rasterio.open(fon / f"{rid}_mb.tif") as g:
        prof, tr, (H, W) = g.profile, g.transform, g.shape
        mb = g.read(1)
    tiles = []
    for t in sorted((emb / rid).glob("*.tif")):
        s = rasterio.open(t)
        tiles.append((s, round((tr.f - s.bounds.top) / 10), round((s.bounds.left - tr.c) / 10)))
    prob = np.zeros((H, W), np.uint8); valid = np.zeros((H, W), bool)
    for r0 in range(0, H, B):
        for c0 in range(0, W, B):
            R0, C0 = max(0, r0 - M), max(0, c0 - M)
            R1, C1 = min(H, r0 + B + M), min(W, c0 + B + M)
            a = np.zeros((64, R1 - R0, C1 - C0), np.uint8)
            for s, tr0, tc0 in tiles:
                ir0, ic0 = max(R0, tr0), max(C0, tc0)
                ir1, ic1 = min(R1, tr0 + s.height), min(C1, tc0 + s.width)
                if ir0 < ir1 and ic0 < ic1:
                    a[:, ir0 - R0:ir1 - R0, ic0 - C0:ic1 - C0] = s.read(window=Window(ic0 - tc0, ir0 - tr0, ic1 - ic0, ir1 - ir0))
            if not a[0].any():
                continue
            p, v = prever(a)
            hh, ww = min(B, H - r0), min(B, W - c0)
            prob[r0:r0 + hh, c0:c0 + ww] = p[r0 - R0:r0 - R0 + hh, c0 - C0:c0 - C0 + ww]
            valid[r0:r0 + hh, c0:c0 + ww] = v[r0 - R0:r0 - R0 + hh, c0 - C0:c0 - C0 + ww]
    for s, _, _ in tiles:
        s.close()
    mm = rasterize([(g, 1) for g in mun[mun[colc].astype(str) == rid.split("_")[0]].to_crs(crs).geometry],
                   out_shape=(H, W), transform=tr, fill=0).astype(bool)
    cl = (prob >= thr * 100) & ~np.isin(mb, c["rotulos"]["classes_seguras_nao_coco_mb"])
    lab, _ = ndimage.label(cl)
    cl &= (np.bincount(lab.ravel()) >= c["predicao"]["pos_proc_min_ha"] * 100)[lab]
    cls = np.where(mm & valid, cl.astype(np.uint8), 255).astype(np.uint8)
    prof.update(nodata=255, compress="deflate", tiled=True)
    with rasterio.open(fp, "w", **prof) as d:
        d.write(np.where(mm & valid, prob, 255).astype(np.uint8), 1)
    with rasterio.open(fc, "w", **prof) as d:
        d.write(cls, 1)
    print(f"{rid}: coco {(cls == 1).sum() / 100:,.0f} ha", flush=True)
