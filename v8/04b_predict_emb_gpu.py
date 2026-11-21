"""04b (v8) — Mesma predição do 04, com decodificação, normalização e janelas na GPU.

Motivo: no 04 a GPU ficava ~7% ocupada (empilhar janelas e normalizar no numpy, 5× por modelo).
Aqui o bloco uint8 vai inteiro para a GPU; cada lote de janelas é decodificado
((v-128)/127), normalizado (robust p2/p98 de cada fold) e inferido em fp32. Matemática idêntica ao 04.
Retomável: pula regiões que já têm _prob.tif e preserva reports/area_caju_2025.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.features import geometry_mask
from rasterio.windows import Window

ROOT = Path(__file__).parent
V7 = ROOT.parent / "v7"
sys.path.insert(0, str(V7))
sys.path.insert(0, str(ROOT))
from src.config import load_config  # noqa: E402
from src.train_utils import get_device  # noqa: E402
import importlib.util  # noqa: E402

_s = importlib.util.spec_from_file_location("p04", ROOT / "04_predict_emb.py")
p04 = importlib.util.module_from_spec(_s); _s.loader.exec_module(p04)


def gauss(size, dev):
    ax = torch.linspace(-1, 1, size, device=dev)
    g = torch.exp(-(ax ** 2) / (2 * 0.25 ** 2))
    w = torch.outer(g, g)
    return w / w.max()


def coords_for(h, w, size, stride):
    rs = list(range(0, max(1, h - size + 1), stride))
    cs = list(range(0, max(1, w - size + 1), stride))
    c = [(r, q) for r in rs for q in cs]
    if (h - size) % stride:
        c += [(h - size, q) for q in cs]
    if (w - size) % stride:
        c += [(r, w - size) for r in rs]
    return c


@torch.no_grad()
def block_prob(xu8: np.ndarray, members, cfg, dev, bs=32):
    size = cfg["inference"]["chip_size"]
    stride = size - cfg["inference"]["overlap"]
    x = torch.from_numpy(xu8).to(dev)                     # (C,h,w) uint8
    _, h, w = x.shape
    valid = (x != 0).all(0)
    if h < size or w < size:                               # borda menor que a janela
        pad = (0, max(0, size - w), 0, max(0, size - h))
        x = torch.nn.functional.pad(x, pad)
    H, W = x.shape[1:]
    co = coords_for(H, W, size, stride)
    wk = gauss(size, dev)
    out = torch.zeros((H, W), device=dev)
    for m, st, _ in members:
        lo = torch.tensor([s["lo"] for s in st["per_channel"]], device=dev)[:, None, None]
        hi = torch.tensor([s["hi"] for s in st["per_channel"]], device=dev)[:, None, None]
        acc = torch.zeros((H, W), device=dev); wac = torch.zeros((H, W), device=dev)
        for i in range(0, len(co), bs):
            cc = co[i:i + bs]
            b = torch.stack([x[:, r:r + size, q:q + size] for r, q in cc]).float()
            b = ((b - 128.0) / 127.0 - lo) / (hi - lo)
            p = torch.sigmoid(m(b.clamp_(0, 1))).squeeze(1)
            for (r, q), pp in zip(cc, p):
                acc[r:r + size, q:q + size] += pp * wk
                wac[r:r + size, q:q + size] += wk
        out += acc / wac.clamp_min(1e-6)
    prob = (out / len(members))[:h, :w]
    prob[~valid] = float("nan")
    return prob.cpu().numpy()


def predict_region(vrt: Path, members, cfg, dev, block=2048, margin=128):
    rid = vrt.stem
    od = ROOT / "predictions" / "2025"
    thr = float(np.mean([t for *_, t in members]))
    with rasterio.open(vrt) as src:
        H, W, tr, crs = src.height, src.width, src.transform, src.crs
        prob = np.full((H, W), 255, dtype="uint8")
        for r0 in range(0, H, block):
            for c0 in range(0, W, block):
                rr, cc = max(0, r0 - margin), max(0, c0 - margin)
                win = Window(cc, rr, min(W, c0 + block + margin) - cc, min(H, r0 + block + margin) - rr)
                raw = src.read(window=win)
                if not (raw != 0).all(axis=0).any():
                    continue
                p = block_prob(raw, members, cfg, dev)
                pr = np.where(np.isfinite(p), np.clip(np.round(p * 100), 0, 100), 255)
                ir, ic = r0 - rr, c0 - cc
                hh, ww = min(block, H - r0), min(block, W - c0)
                prob[r0:r0 + hh, c0:c0 + ww] = pr[ir:ir + hh, ic:ic + ww]
    if rid.startswith("m"):
        import geopandas as gpd
        g = gpd.read_file(ROOT.parent / "dados_externos" / "municipios25" / "municipios25.gpkg")
        g = g[g["cod"].astype(str) == rid[1:8]].to_crs(crs)
        if len(g):
            prob[geometry_mask(g.geometry, (H, W), tr)] = 255
    cls = np.where(prob == 255, 255, (prob >= thr * 100).astype("uint8")).astype("uint8")
    prof = dict(driver="GTiff", height=H, width=W, count=1, dtype="uint8", crs=crs, transform=tr,
                nodata=255, tiled=True, blockxsize=512, blockysize=512, compress="deflate")
    for arr, suf in [(prob, "prob"), (cls, "class")]:
        with rasterio.open(od / f"{rid}_{suf}.tif", "w", **prof) as d:
            d.write(arr, 1)
    return {"regiao": rid, "caju_ha": round(float((cls == 1).sum()) * 100 / 1e4, 1), "limiar": round(thr, 3)}


def main():
    cfg = load_config(str(V7 / "config" / "config.yaml"))
    cfg["experiment"]["channels"] = "emb"
    dev = get_device()
    members = p04.load_members(cfg, dev)
    rep = ROOT / "reports" / "area_caju_2025.json"
    res = json.loads(rep.read_text(encoding="utf-8")) if rep.exists() else []
    done = {r["regiao"] for r in res}
    for v in sorted((ROOT / "data" / "ee_exports" / "2025" / "embeddings").glob("m*.vrt")):
        if v.stem in done and (ROOT / "predictions" / "2025" / f"{v.stem}_prob.tif").exists():
            continue
        r = predict_region(v, members, cfg, dev)
        print(r, flush=True)
        res.append(r)
        rep.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
