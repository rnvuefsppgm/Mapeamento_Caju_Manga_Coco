"""02 (v9) — Chips SÓ embeddings 2025 (64 canais): campo (Serra do Mel) + pseudo (outros municípios).

Campo: polígonos -> 1; faixa de exclusão (ignore) de `v9.ignore_buffer_m` em volta; resto 0.
Pseudo (municípios da v8 aprovados no filtro mapa/PAM, exceto Serra do Mel), duas variantes nos MESMOS chips:
  A = só negativos MapBiomas (classes em NEG_MB) ; resto ignorado            -> testa "negativos"
  B = A + caju do modelo v8 (prob >= pos_thr, manchas >= min_px, fora de NEG_MB)
        + não-caju do modelo (prob <= neg_thr) ; zona incerta ignorada          -> testa "pseudo-rótulos"
Saídas: metadata/chips_metadata_all.parquet (+ _A / _B após o cache), chips .npz float16.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from scipy import ndimage

from src.chip_generator import _spatial_block_id, generate_chips, save_metadata
from src.config import ensure_dirs, load_config, resolve_path, setup_logging
from src.geo_utils import read_vector
from src.labels import rasterize_labels

ROOT = Path(__file__).parent
V7, V8 = ROOT.parent / "v7", ROOT.parent / "v8"
# floresta, mangue, praia/duna, urbano, outras não vegetadas, mineração, aquicultura, rio/lago
NEG_MB = [3, 5, 23, 24, 25, 30, 31, 33]


def dec_v7(a):  # int8 (v7): x*127 ; -128 = NoData
    f = a.astype("float32") / 127.0
    f[a == -128] = np.nan
    return f


def dec_v8(a):  # uint8 (v8): x*127+128 ; 0 = NoData
    f = (a.astype("float32") - 128.0) / 127.0
    f[a == 0] = np.nan
    return f


def field_chips(cfg, log):
    gdf = read_vector(str(V7 / "data" / "vectors" / "Pol_Caju.shp"), cfg["project"]["crs_working"])
    buf = float(cfg["v9"]["ignore_buffer_m"])
    recs = []
    for emb in sorted((V7 / "data" / "embeddings").glob("emb_features_*_2025.tif")):
        with rasterio.open(emb) as d:
            feats, tr, sh = dec_v7(d.read()), d.transform, (d.height, d.width)
        lab = rasterize_labels(gdf, sh, tr, cfg)
        ring = rasterize([(g, 1) for g in gdf.buffer(buf).geometry], out_shape=sh,
                         transform=tr, fill=0).astype(bool)
        lab[ring & (lab == 0)] = 255
        lab[~np.isfinite(feats).all(axis=0)] = 255
        recs += generate_chips(feats, lab, cfg, resolve_path(cfg, "chips"),
                               prefix="campo_" + emb.stem.replace("emb_features_", ""),
                               meta_common={"municipality": "Serra do Mel", "year": 2025,
                                            "source": "field", "crs": cfg["project"]["crs_working"]},
                               transform=tr)
        del feats, lab
    df = pd.DataFrame([r.__dict__ for r in recs]).drop(columns=["extra"], errors="ignore")
    df["variant"] = "field"
    log.info("Campo: %d chips (%d com caju).", len(df), int((~df.is_negative).sum()))
    return df


def pseudo_chips(cfg, log):
    c = cfg["v9"]
    rep = pd.DataFrame(json.loads((V8 / "reports" / "area_caju_2025.json").read_text(encoding="utf-8")))
    rep = rep[rep.regiao.str.startswith("m")].copy()
    rep["cod"] = rep.regiao.str[1:8].astype(int)
    pam = pd.read_csv(ROOT.parent / "dados_externos" / "ranking_caju_caatinga_2023_2025.csv")
    rep = rep.merge(pam[["cod", "area_ha_2025"]], on="cod", how="left")
    rep["razao_mapa_pam"] = rep.caju_ha / rep.area_ha_2025.replace(0, np.nan)
    lo, hi = c["pam_ratio"]
    rep["aceito"] = rep.razao_mapa_pam.between(lo, hi) & (rep.cod != 2413359)  # sem Serra do Mel
    rep.to_csv(ROOT / "reports" / "pseudo_filtro_municipios.csv", index=False)
    ok = rep[rep.aceito]
    log.info("Municípios aceitos p/ pseudo: %d de %d (razão mapa/PAM em [%s, %s]).",
             len(ok), len(rep), lo, hi)
    if ok.empty:
        return pd.DataFrame()
    per = max(2, int(c["max_pseudo_chips"]) // len(ok))
    size, rng = cfg["chips"]["size"], np.random.default_rng(42)
    out_dir = resolve_path(cfg, "chips")
    rows = []
    for rid in ok.regiao:
        e = rasterio.open(V8 / "data" / "ee_exports" / "2025" / "embeddings" / f"{rid}.vrt")
        with rasterio.open(V8 / "predictions" / "2025" / f"{rid}_prob.tif") as p:
            P = p.read(1)
        mb = sorted((ROOT / "data" / "mapbiomas_api").glob(f"{rid}_*.tif"))  # API MapBiomas (sem GEE)
        M = np.zeros(P.shape, dtype="uint8")
        if mb:
            with rasterio.open(mb[-1]) as m:
                reproject(rasterio.band(m, 1), M, dst_transform=e.transform, dst_crs=e.crs,
                          dst_nodata=0, resampling=Resampling.nearest)
        else:
            log.warning("%s: sem MapBiomas; só pseudo-rótulos do modelo", rid)
        inside = P != 255
        negmb = np.isin(M, NEG_MB) & inside
        pos = (P >= c["pos_thr"] * 100) & inside & ~negmb
        lab_cc, n = ndimage.label(pos)
        if n:
            sizes = ndimage.sum(pos, lab_cc, range(1, n + 1))
            pos = np.isin(lab_cc, np.where(sizes >= c["min_px"])[0] + 1)
        LA = np.full(P.shape, 255, "uint8"); LA[negmb] = 0
        LB = LA.copy(); LB[(P <= c["neg_thr"] * 100) & inside] = 0; LB[pos] = 1
        cand_pos, cand_neg = [], []
        for r0 in range(0, P.shape[0] - size + 1, size):
            for c0 in range(0, P.shape[1] - size + 1, size):
                w = LB[r0:r0 + size, c0:c0 + size]
                if (w != 255).mean() < c["min_confident_fraction"]:
                    continue
                (cand_pos if (w == 1).mean() >= 0.02 else cand_neg).append((r0, c0))
        pick = [cand_pos[i] for i in rng.permutation(len(cand_pos))[:per // 2]]
        pick += [cand_neg[i] for i in rng.permutation(len(cand_neg))[:per - len(pick)]]
        for k, (r0, c0) in enumerate(pick):
            f = dec_v8(e.read(window=Window(c0, r0, size, size)))
            valid = np.isfinite(f).all(axis=0)
            f16 = np.nan_to_num(f, nan=0.0).astype("float16")
            bid = _spatial_block_id(e.transform, r0, c0, size, cfg["split"]["block_size_km"] * 1000.0)
            for var, L in (("A", LA), ("B", LB)):
                lab = L[r0:r0 + size, c0:c0 + size].copy()
                lab[~valid] = 255
                cid = f"ps{var}_{rid}_{k:04d}"
                path = out_dir / f"{cid}.npz"
                np.savez(path, features=f16, label=lab, valid=valid.astype("uint8"))
                rows.append({"chip_id": cid, "path": str(path), "municipality": rid,
                             "valid_fraction": float(valid.mean()),
                             "cashew_fraction": float((lab == 1).mean()),
                             "background_fraction": float((lab == 0).mean()),
                             "n_channels": 64, "chip_size": size, "resolution": 10,
                             "crs": cfg["project"]["crs_working"], "block_id": bid,
                             "is_negative": bool((lab == 1).sum() == 0), "year": 2025,
                             "source": "pseudo", "variant": var})
        e.close()
        log.info("%s: %d chips (candidatos pos=%d neg=%d)", rid, len(pick), len(cand_pos), len(cand_neg))
    return pd.DataFrame(rows)


def main():
    cfg = load_config(None)
    ensure_dirs(cfg)
    log = setup_logging("02_chips_v9", cfg)
    df = pd.concat([field_chips(cfg, log), pseudo_chips(cfg, log)], ignore_index=True)
    df.to_parquet(resolve_path(cfg, "metadata") / "chips_metadata_all.parquet", index=False)
    log.info("Total: %d chips | %s", len(df), df.groupby("variant").size().to_dict())


if __name__ == "__main__":
    main()
