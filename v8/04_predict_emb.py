"""04 (v8) — Mapa de caju 2025 por município com o ensemble `unet_emb` da v7 (só embeddings).

Entrada: data/ee_exports/2025/embeddings/<regiao>.vrt (uint8; (v-128)/127; 0 = NoData).
Modelos: ../v7/models/ensemble_unet_emb/fold*.pt (normalização e limiar de cada fold).
Processa em BLOCOS (padrão 2048 px + margem 128) para caber na RAM em municípios grandes.
Saída por região: predictions/2025/<regiao>_prob.tif (uint8 0–100; 255 = NoData),
<regiao>_class.tif (0/1; 255) — recortados ao polígono do município (sem buffer).

    python 04_predict_emb.py [--regions m2413359_SerradoMel ...] [--block 2048]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window

ROOT = Path(__file__).parent
V7 = ROOT.parent / "v7"
sys.path.insert(0, str(V7))
from src.config import load_config  # noqa: E402
from src.dataset import apply_norm  # noqa: E402
from src.inference_utils import sliding_window_predict  # noqa: E402
from src.models import build_model  # noqa: E402
from src.train_utils import get_device  # noqa: E402


def load_members(cfg, device, name="unet_emb"):
    import torch
    mem = []
    for f in sorted((V7 / "models" / f"ensemble_{name}").glob("fold*.pt")):
        ck = torch.load(f, map_location="cpu", weights_only=False)
        m = build_model(cfg, in_channels=len(ck["stats"]["per_channel"]))
        m.load_state_dict(ck["model_state"])
        mem.append((m.to(device).eval(), ck["stats"], float(ck["threshold"])))
    return mem


def predict_region(vrt: Path, members, cfg, device, block: int, margin: int = 128):
    rid = vrt.stem
    out_dir = ROOT / "predictions" / "2025"
    out_dir.mkdir(parents=True, exist_ok=True)
    thr = float(np.mean([t for *_, t in members]))
    with rasterio.open(vrt) as src:
        H, W, tr, crs = src.height, src.width, src.transform, src.crs
        prob = np.full((H, W), 255, dtype="uint8")
        for r0 in range(0, H, block):
            for c0 in range(0, W, block):
                rr, cc = max(0, r0 - margin), max(0, c0 - margin)
                win = Window(cc, rr, min(W, c0 + block + margin) - cc, min(H, r0 + block + margin) - rr)
                raw = src.read(window=win)
                valid = (raw != 0).all(axis=0)
                if not valid.any():
                    continue
                x = (raw.astype("float32") - 128.0) / 127.0
                p = None
                for m, st, _ in members:
                    q = sliding_window_predict(apply_norm(x, st), valid, m, cfg, device)
                    p = q if p is None else p + q
                p = p / len(members)
                pr = np.where(np.isfinite(p) & valid, np.clip(np.round(p * 100), 0, 100), 255)
                ir, ic = r0 - rr, c0 - cc
                hh, ww = min(block, H - r0), min(block, W - c0)
                prob[r0:r0 + hh, c0:c0 + ww] = pr[ir:ir + hh, ic:ic + ww]
    # recorte ao município (polígono sem buffer, se existir)
    gpk = ROOT.parent / "dados_externos" / "municipios25" / "municipios25.gpkg"
    if rid.startswith("m"):
        import geopandas as gpd
        g = gpd.read_file(gpk)
        g = g[g["cod"].astype(str) == rid[1:8]].to_crs(crs)
        if len(g):
            outside = geometry_mask(g.geometry, (H, W), tr)
            prob[outside] = 255
    cls = np.where(prob == 255, 255, (prob >= thr * 100).astype("uint8")).astype("uint8")
    prof = dict(driver="GTiff", height=H, width=W, count=1, dtype="uint8", crs=crs, transform=tr,
                nodata=255, tiled=True, blockxsize=512, blockysize=512, compress="deflate")
    for arr, suf in [(prob, "prob"), (cls, "class")]:
        with rasterio.open(out_dir / f"{rid}_{suf}.tif", "w", **prof) as d:
            d.write(arr, 1)
    area = float((cls == 1).sum()) * 100 / 1e4  # ha (pixel 10 m)
    return {"regiao": rid, "caju_ha": round(area, 1), "limiar": round(thr, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="*", default=None)
    ap.add_argument("--block", type=int, default=2048)
    a = ap.parse_args()
    cfg = load_config(str(V7 / "config" / "config.yaml"))
    cfg["experiment"]["channels"] = "emb"
    device = get_device()
    members = load_members(cfg, device)
    print(f"{len(members)} modelos | device={device}")
    base = ROOT / "data" / "ee_exports" / "2025" / "embeddings"
    vrts = sorted(base.glob("*.vrt"))
    if a.regions:
        vrts = [v for v in vrts if v.stem in set(a.regions)]
    res = []
    rep = ROOT / "reports" / "area_caju_2025.json"
    for v in vrts:
        r = predict_region(v, members, cfg, device, a.block)
        print(r, flush=True)
        res.append(r)
        rep.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
