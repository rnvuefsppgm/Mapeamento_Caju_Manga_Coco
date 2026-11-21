"""Dataset PyTorch, split espacial, normalização (só-treino) e augmentation.

Pontos metodológicos:
    * SPLIT ESPACIAL por ``block_id`` (ETAPAS 21-22): chips do mesmo bloco nunca
      caem em subconjuntos diferentes -> previne spatial/polygon/scene leakage.
      Alternativa Leave-One-Municipality-Out também suportada.
    * NORMALIZAÇÃO estimada EXCLUSIVAMENTE no treino (ETAPA 28): média/desvio ou
      percentis robustos calculados só sobre chips de treino e reaplicados a
      val/test/inferência. Nunca se usa estatística do teste.
    * AUGMENTATION geométrica aplicada IGUALMENTE a features e label; nunca se
      aplica perturbação radiométrica arbitrária ao SAR (preserva significado
      físico do retroespalhamento) — só flips/rot90 (ETAPA 23).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import Dataset
    _TORCH = True
except Exception:  # pragma: no cover
    _TORCH = False
    Dataset = object  # type: ignore


# --------------------------------------------------------------------------- #
# Split espacial
# --------------------------------------------------------------------------- #
def spatial_split(
    meta: pd.DataFrame, cfg: Dict[str, Any], seed: int = 42
) -> Dict[str, List[str]]:
    """Divide chips em treino/val/teste respeitando grupos espaciais.

    Returns:
        {"train": [chip_id...], "val": [...], "test": [...]}.
    """
    strategy = cfg["split"]["strategy"]
    rng = np.random.default_rng(seed)
    full_meta = meta
    meta = field_meta(meta)  # v6: grupos/val/teste só com rótulos de campo

    if strategy == "leave_municipality_out":
        group_field = cfg["split"]["group_field"]
        groups = meta[group_field].dropna().unique().tolist()
    else:
        groups = sorted(meta["block_id"].dropna().unique().tolist())
        group_field = "block_id"

    rng.shuffle(groups)
    n = len(groups)
    f = cfg["split"]["fractions"]
    n_train = int(round(f["train"] * n))
    n_val = int(round(f["val"] * n))
    train_g = set(groups[:n_train])
    val_g = set(groups[n_train:n_train + n_val])
    test_g = set(groups[n_train + n_val:])

    def ids_for(gset: set) -> List[str]:
        return meta.loc[meta[group_field].isin(gset), "chip_id"].tolist()

    split = {"train": ids_for(train_g), "val": ids_for(val_g), "test": ids_for(test_g)}
    split["train"] += pseudo_ids_for_train(full_meta, val_g | test_g)
    logger.info("Split espacial (%s): train=%d val=%d test=%d chips | grupos=%d.",
                strategy, len(split["train"]), len(split["val"]),
                len(split["test"]), n)
    return split


_MM: Dict[str, Any] = {}


def load_chip(row, cfg: Dict[str, Any]):
    """v7: (features float32 só com os canais do modelo, label int64).

    Usa o cache memmap (02c) quando ``row`` tem ``cache_idx``; senão lê o .npz.
    O memmap é aberto uma vez por processo (worker) e lê só os canais pedidos.
    """
    idx = row.get("cache_idx") if hasattr(row, "get") else None
    if idx is not None and not pd.isna(idx):
        d = Path(cfg["_output_root"]) / cfg["paths"]["chips"] / ".." / "cache"
        if "f" not in _MM:
            _MM["f"] = np.load(d / "features.npy", mmap_mode="r")
            _MM["l"] = np.load(d / "labels.npy", mmap_mode="r")
        feats = select_channels(_MM["f"][int(idx)], cfg)
        return np.asarray(feats, dtype="float32"), np.asarray(_MM["l"][int(idx)], dtype="int64")
    data = np.load(row["path"])
    return (select_channels(data["features"], cfg).astype("float32"),
            data["label"].astype("int64"))


def select_channels(arr: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """v7: escolhe os canais do modelo. Chips guardam [Sentinel ; embeddings].

    experiment.channels: "sentinel" (só Sentinel), "emb" (só embeddings),
    "both" (todos). Os últimos ``embeddings.n_bands`` canais são os embeddings.
    """
    mode = cfg.get("experiment", {}).get("channels", "both")
    if mode == "both" or not cfg.get("embeddings", {}).get("enabled"):
        return arr
    nb = int(cfg["embeddings"]["n_bands"])
    return arr[-nb:] if mode == "emb" else arr[:-nb]


def field_meta(meta: pd.DataFrame) -> pd.DataFrame:
    """v6: apenas chips com rótulo de campo (exclui pseudo-rótulos)."""
    if "source" not in meta.columns:
        return meta
    return meta[meta["source"].fillna("field") == "field"]


def pseudo_ids_for_train(meta: pd.DataFrame, holdout_blocks) -> List[str]:
    """v6: pseudo-chips entram SÓ no treino e nunca em blocos de val/teste.

    Também descarta pseudo-chips de blocos vizinhos (8-conectados) aos blocos
    retidos, reduzindo vazamento por autocorrelação espacial.
    """
    if "source" not in meta.columns:
        return []
    ps = meta[meta["source"] == "pseudo"]
    if ps.empty:
        return []
    banned = set()
    for b in holdout_blocks:
        try:
            _, ix, iy = str(b).rsplit("_", 2)
            ix, iy = int(ix), int(iy)
            prefix = str(b).rsplit("_", 2)[0]
            banned |= {f"{prefix}_{ix + dx}_{iy + dy}"
                       for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
        except ValueError:
            banned.add(str(b))
    keep = ps[~ps["block_id"].astype(str).isin(banned)]
    logger.info("Pseudo-chips no treino: %d de %d (descartados %d perto de val/teste).",
                len(keep), len(ps), len(ps) - len(keep))
    return keep["chip_id"].tolist()


# --------------------------------------------------------------------------- #
# Normalização (estimada só no treino)
# --------------------------------------------------------------------------- #
def compute_norm_stats(
    meta: pd.DataFrame, train_ids: List[str], cfg: Dict[str, Any],
    max_chips: int = 200,
) -> Dict[str, Any]:
    """Estima estatísticas de normalização por canal usando SÓ chips de treino.

    Para método 'robust': percentis (p2,p98) por canal. Para 'zscore': média/std.
    """
    method = cfg["normalization"]["method"]
    lo_p, hi_p = cfg["normalization"]["clip_percentiles"]
    train_meta = meta[meta["chip_id"].isin(train_ids)]
    sample = train_meta.sample(min(max_chips, len(train_meta)), random_state=42)

    acc: Optional[List[List[float]]] = None
    # Barra de progresso: leitura dos chips de treino p/ estimar as estatísticas.
    for _, srow in tqdm(sample.iterrows(), total=len(sample),
                        desc="Estatísticas de normalização", unit="chip", leave=False):
        arr = load_chip(srow, cfg)[0]  # v7: cache memmap + seleção de canais
        c = arr.shape[0]
        if acc is None:
            acc = [[] for _ in range(c)]
        flat = arr.reshape(c, -1)
        for ci in range(c):
            vals = flat[ci]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                acc[ci].append(vals[::7])  # subamostra p/ memória

    stats: Dict[str, Any] = {"method": method, "per_channel": []}
    for ci, chunks in enumerate(acc or []):
        vals = np.concatenate(chunks) if chunks else np.array([0.0, 1.0])
        if method == "robust":
            lo = float(np.percentile(vals, lo_p))
            hi = float(np.percentile(vals, hi_p))
            stats["per_channel"].append({"lo": lo, "hi": max(hi, lo + 1e-6)})
        else:  # zscore
            mu = float(vals.mean())
            sd = float(vals.std() + 1e-6)
            stats["per_channel"].append({"mean": mu, "std": sd})

    out = Path(cfg["_output_root"]) / cfg["normalization"]["stats_file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Estatísticas de normalização (%s) salvas: %s (%d canais).",
                method, out.name, len(stats["per_channel"]))
    return stats


def apply_norm(arr: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    """Aplica normalização por canal a um array (C,H,W)."""
    method = stats["method"]
    out = arr.astype("float32").copy()
    for ci, s in enumerate(stats["per_channel"]):
        if ci >= out.shape[0]:
            break
        if method == "robust":
            out[ci] = np.clip((out[ci] - s["lo"]) / (s["hi"] - s["lo"]), 0.0, 1.0)
        else:
            out[ci] = (out[ci] - s["mean"]) / s["std"]
    return out


# --------------------------------------------------------------------------- #
# Augmentation (geométrica, aplicada a features + label)
# --------------------------------------------------------------------------- #
def _augment(features: np.ndarray, label: np.ndarray, rng: np.random.Generator):
    """Flips e rot90 aplicados de forma idêntica a features e label."""
    if rng.random() < 0.5:  # horizontal flip
        features = features[:, :, ::-1]
        label = label[:, ::-1]
    if rng.random() < 0.5:  # vertical flip
        features = features[:, ::-1, :]
        label = label[::-1, :]
    k = int(rng.integers(0, 4))  # rot90 {0,90,180,270}
    if k:
        features = np.rot90(features, k=k, axes=(1, 2))
        label = np.rot90(label, k=k)
    return np.ascontiguousarray(features), np.ascontiguousarray(label)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class CashewChipDataset(Dataset):  # type: ignore
    """Dataset de chips para segmentação binária de caju."""

    def __init__(
        self,
        meta: pd.DataFrame,
        chip_ids: List[str],
        stats: Dict[str, Any],
        cfg: Dict[str, Any],
        augment: bool = False,
    ) -> None:
        if not _TORCH:
            raise RuntimeError("PyTorch não está instalado.")
        self.meta = meta.set_index("chip_id").loc[chip_ids].reset_index()
        self.stats = stats
        self.cfg = cfg
        self.augment = augment
        self.ignore_val = cfg["chips"]["label_ignore_value"]
        self.rng = np.random.default_rng(cfg["project"]["random_seed"])

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int):
        row = self.meta.iloc[idx]
        features, label = load_chip(row, self.cfg)  # v7: cache memmap

        features = apply_norm(features, self.stats)
        if self.augment:
            features, label = _augment(features, label, self.rng)

        x = torch.from_numpy(features)
        y = torch.from_numpy(label.copy())
        return x, y
