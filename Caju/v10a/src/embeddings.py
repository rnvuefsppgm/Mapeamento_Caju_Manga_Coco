"""v7 — Google Satellite Embedding (AlphaEarth) alinhado à grade dos COGs Sentinel.

Download SEM tarefas no Earth Engine: ``ee.data.computePixels`` em tiles (síncrono),
na MESMA grade (CRS + transform + dimensões) do COG de features Sentinel, para
que os chips de embeddings e de Sentinel fiquem pixel a pixel alinhados.
Armazenamento em int8 (valor*127; -128 = NoData) = 64 B/pixel.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

import numpy as np
import rasterio

logger = logging.getLogger(__name__)
NODATA_I8 = -128


def emb_path_for(cog: str | Path, cfg: Dict[str, Any]) -> Path:
    """Caminho do GeoTIFF de embeddings correspondente a um COG de features."""
    from .config import resolve_path
    d = Path(cfg["_output_root"]) / cfg["embeddings"]["dir"]
    return d / f"emb_{Path(cog).stem}.tif"


def _ee_image(cfg: Dict[str, Any]):
    import ee
    e = cfg["embeddings"]
    y = int(e["year"])
    img = (ee.ImageCollection(e["collection"])
           .filterDate(f"{y}-01-01", f"{y + 1}-01-01").mosaic())
    sc = float(e["scale"])
    # NoData -> -128 ; valores [-1,1] -> [-127,127]
    return img.multiply(sc).round().clamp(-127, 127).unmask(NODATA_I8).toInt8()


def download_embeddings_for(cog: str | Path, cfg: Dict[str, Any]) -> Path:
    """Baixa os embeddings na grade do ``cog`` (retomável: pula se já existe)."""
    import ee
    out = emb_path_for(cog, cfg)
    if out.exists():
        logger.info("já existe: %s", out.name)
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(cog) as ds:
        tr, H, W, crs = ds.transform, ds.height, ds.width, ds.crs.to_string()
    e = cfg["embeddings"]
    nb, tile = int(e["n_bands"]), int(e["tile_px"])
    img = _ee_image(cfg)
    arr = np.full((nb, H, W), NODATA_I8, dtype="int8")
    jobs = [(r, c) for r in range(0, H, tile) for c in range(0, W, tile)]

    def fetch(r, c):
        h, w = min(tile, H - r), min(tile, W - c)
        x0, y0 = tr * (c, r)  # canto superior esquerdo do tile
        req = {"expression": img, "fileFormat": "NUMPY_NDARRAY",
               "grid": {"dimensions": {"width": w, "height": h},
                        "affineTransform": {"scaleX": tr.a, "shearX": tr.b, "translateX": x0,
                                            "shearY": tr.d, "scaleY": tr.e, "translateY": y0},
                        "crsCode": crs}}
        for k in range(6):
            try:
                a = ee.data.computePixels(req)
                return r, c, np.stack([a[n] for n in a.dtype.names])  # (B,h,w)
            except Exception as exc:  # cota/erro transitório -> espera e repete
                time.sleep(5 * 2 ** k)
                err = exc
        raise RuntimeError(f"tile {r},{c} falhou: {err}")

    t0 = time.time()
    with ThreadPoolExecutor(int(e["workers"])) as ex:
        futs = [ex.submit(fetch, r, c) for r, c in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            r, c, a = f.result()
            arr[:, r:r + a.shape[1], c:c + a.shape[2]] = a
            if i % 20 == 0 or i == len(jobs):
                logger.info("  %s: %d/%d tiles (%.0fs)", out.name, i, len(jobs), time.time() - t0)
    tmp = out.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", driver="GTiff", height=H, width=W, count=nb, dtype="int8",
                       crs=crs, transform=tr, nodata=NODATA_I8, tiled=True,
                       blockxsize=256, blockysize=256, compress="deflate") as dst:
        dst.write(arr)
    tmp.replace(out)
    logger.info("salvo %s (%dx%d, %.0fs)", out.name, H, W, time.time() - t0)
    return out


def load_embeddings_for(cog: str | Path, cfg: Dict[str, Any]) -> np.ndarray:
    """Lê os embeddings do ``cog`` como float32 (B,H,W) em [-1,1]; NoData -> NaN."""
    with rasterio.open(emb_path_for(cog, cfg)) as ds:
        a = ds.read()
    f = a.astype("float32") / float(cfg["embeddings"]["scale"])
    f[a == NODATA_I8] = np.nan
    return f


def stack_with_embeddings(features: np.ndarray, cog: str | Path,
                          cfg: Dict[str, Any]) -> np.ndarray:
    """Concatena [Sentinel ; embeddings] se ``embeddings.enabled``."""
    if not cfg.get("embeddings", {}).get("enabled"):
        return features
    emb = load_embeddings_for(cog, cfg)
    if emb.shape[1:] != features.shape[1:]:
        raise ValueError(f"grade diferente: {emb.shape} vs {features.shape} ({cog})")
    return np.concatenate([features, emb], axis=0)
