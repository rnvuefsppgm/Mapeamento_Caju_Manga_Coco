"""11 (v1) — Chips + cache 8 bits para a rede (formato do caju v10b: dados/cache/features_u8.npy, labels.npy).

campo  (source=field): chips do 08 (stride 64) em Livramento; rótulo = só o conferido: 1 = polígonos (código 1);
       0 = códigos 3/4 onde a A5 não vê manga (conf=0 e prob<50); resto 255. É o que AVALIA a rede no CV.
conc   (source=pseudo): rótulos v1 (10_rotulos_chapada.py) dos 4 municípios, 256 px sem sobreposição,
       até 30 com manga + 30 sem; em Livramento fora da área dos chips de campo. Só treino.
caju   (source=pseudo): proteção 2 — chips de campo do caju (Serra do Mel), todo pixel rotulado vira NÃO-manga.
Saídas: dados/chips/{conc,caju}_*.npz, dados/cache/*, metadata/chips_metadata_v1.parquet (<= 10 linhas no log).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from src.chip_generator import _spatial_block_id
from src.config import load_config, resolve_path, setup_logging

cfg = load_config(Path(__file__).parent / "config/config.yaml")
log = setup_logging("11_chips_v1", cfg)
T = cfg["treino_v1"]; S = cfg["chips"]["size"]; BK = cfg["manga"]["chips_campo"]["bloco_cv_km"] * 1000.0
D = Path(cfg["_output_root"]) / "dados"; CH = D / "chips"
rng = np.random.default_rng(7)
rows = []

# ---- campo (rótulo conferido) ----
fm = pd.read_parquet(resolve_path(cfg, "metadata") / "chips_metadata_campo.parquet")
lab_campo = {}
for rid, sub in fm.groupby("municipality"):
    with rasterio.open(D / f"predicoes/xgb_auto/{rid}_prob.tif") as a, rasterio.open(D / f"predicoes/xgb_auto/{rid}_conf.tif") as c:
        PA, CF = a.read(1), c.read(1) == 1
    for r in sub.itertuples():
        z = np.load(r.path)
        k, pa, cf = z["codes"], PA[r.row:r.row + S, r.col:r.col + S], CF[r.row:r.row + S, r.col:r.col + S]
        L = np.full(k.shape, 255, np.uint8)
        L[np.isin(k, [3, 4]) & ~cf & (pa < 50)] = 0
        L[k == 1] = 1
        L[z["valid"] == 0] = 255
        lab_campo[r.chip_id] = L
        rows.append({"chip_id": r.chip_id, "path": r.path, "municipality": rid, "block_id": r.block_id,
                     "source": "field", "variant": "field", "row": r.row, "col": r.col})
campo_box = {rid: (s.row.min(), s.row.max() + S, s.col.min(), s.col.max() + S) for rid, s in fm.groupby("municipality")}

# ---- conc (rótulos v1 dos 4 municípios) ----
rids = {int(r[1:8]): r for r in (D / "aoi/regioes.txt").read_text().split() if r}
for cod in T["municipios"]:
    rid = rids[cod]
    with rasterio.open(D / f"rotulos/{rid}_v1.tif") as d:
        L = d.read(1)
    cp_, cn = [], []
    for r0 in range(0, L.shape[0] - S + 1, S):
        for c0 in range(0, L.shape[1] - S + 1, S):
            b = campo_box.get(rid)
            if b and r0 < b[1] and r0 + S > b[0] and c0 < b[3] and c0 + S > b[2]:
                continue
            w = L[r0:r0 + S, c0:c0 + S]
            if (w != 255).mean() < T["pseudo_min_rotulado"]:
                continue
            (cp_ if (w == 1).mean() >= 0.02 else cn).append((r0, c0))
    pick = [cp_[i] for i in rng.permutation(len(cp_))[:T["pseudo_por_classe"]]] + \
           [cn[i] for i in rng.permutation(len(cn))[:T["pseudo_por_classe"]]]
    with rasterio.open(D / f"embeddings/{rid}.vrt") as e:
        for k, (r0, c0) in enumerate(pick):
            f = e.read(window=Window(c0, r0, S, S)); v = (f != 0).all(axis=0); f[:, ~v] = 128
            lab = L[r0:r0 + S, c0:c0 + S].copy(); lab[~v] = 255
            cid = f"conc_{rid}_{k:03d}"; p = CH / f"{cid}.npz"
            np.savez(p, features=f, label=lab, valid=v.astype("uint8"))
            rows.append({"chip_id": cid, "path": str(p), "municipality": rid, "source": "pseudo", "variant": "conc",
                         "block_id": _spatial_block_id(e.transform, r0, c0, S, BK)})
    log.info(f"{rid}: {len(pick)} chips conc (candidatos com manga {len(cp_)}, sem {len(cn)})")

# ---- caju como não-manga (proteção 2) ----
cj = sorted(Path(T["caju_chips_dir"]).glob("campo_*.npz"))
frac = []
for p in cj:
    with np.load(p) as z:
        frac.append((z["label"] == 1).mean())
sel = [cj[i] for i in np.argsort(frac)[::-1][:T["caju_chips"]]]
for k, p in enumerate(sel):
    with np.load(p) as z:
        x, lab = z["features"].astype(np.float32), z["label"].copy()
        v = z["valid"].astype(bool) if "valid" in z.files else np.isfinite(x).all(axis=0)
    f = np.clip(np.round(np.nan_to_num(x) * 127) + 128, 1, 255).astype(np.uint8); f[:, ~v] = 128
    lab[lab != 255] = 0
    cid = f"caju_{k:03d}"; q = CH / f"{cid}.npz"
    np.savez(q, features=f, label=lab, valid=v.astype("uint8"))
    rows.append({"chip_id": cid, "path": str(q), "municipality": "SerradoMel_caju", "source": "pseudo",
                 "variant": "caju_neg", "block_id": f"caju_{k}"})

# ---- cache ----
meta = pd.DataFrame(rows).reset_index(drop=True)
out = D / "cache"
F = np.lib.format.open_memmap(out / "features_u8.npy", mode="w+", dtype="uint8", shape=(len(meta), 64, S, S))
Lc = np.lib.format.open_memmap(out / "labels.npy", mode="w+", dtype="uint8", shape=(len(meta), S, S))
fr = []
for i, r in meta.iterrows():
    z = np.load(r.path)
    F[i] = z["features"]
    Lc[i] = lab_campo[r.chip_id] if r.source == "field" else z["label"]
    fr.append(((Lc[i] == 1).mean(), (Lc[i] == 0).mean()))
F.flush(); Lc.flush()
meta["cache_idx"] = np.arange(len(meta))
meta["cashew_fraction"], meta["background_fraction"] = zip(*fr)
meta.to_parquet(resolve_path(cfg, "metadata") / "chips_metadata_v1.parquet", index=False)
log.info(f"RESUMO: {meta.groupby('variant').size().to_dict()} | blocos de campo {meta[meta.source == 'field'].block_id.nunique()} | "
         f"manga {100 * meta.cashew_fraction.mean():.1f}% · não {100 * meta.background_fraction.mean():.1f}% dos pixels")
