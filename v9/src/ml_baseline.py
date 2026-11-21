"""Baseline tradicional por pixel: Random Forest (ETAPA 25).

Objetivo: verificar se o Deep Learning realmente supera uma abordagem clássica
(ETAPA 48, questão 7). Para comparação JUSTA, o baseline usa:
    * exatamente os MESMOS chips e o MESMO split espacial;
    * as MESMAS features (canais dos chips) como atributos por pixel;
    * as MESMAS métricas focadas na classe caju.
Diferença: RF classifica pixel-a-pixel, sem contexto espacial (o que é
justamente o que a CNN adiciona).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestClassifier
    _SK = True
except Exception:  # pragma: no cover
    _SK = False


def _load_pixels(
    meta: pd.DataFrame, chip_ids: List[str], ignore_val: int,
    max_pixels_per_chip: int = 2000, seed: int = 42, desc: str = "carregando pixels",
) -> Tuple[np.ndarray, np.ndarray]:
    """Amostra pixels (features, label) dos chips, ignorando pixels 'ignore'."""
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    sub = meta[meta["chip_id"].isin(chip_ids)]
    # Barra de progresso: leitura/amostragem de pixels por chip.
    for path in tqdm(sub["path"], desc=desc, unit="chip", leave=False):
        data = np.load(path)
        feats = data["features"]           # (C,H,W)
        label = data["label"]              # (H,W)
        valid = (label != ignore_val)
        c = feats.shape[0]
        flat_x = feats.reshape(c, -1).T    # (N, C)
        flat_y = label.reshape(-1)
        flat_v = valid.reshape(-1)
        idx = np.where(flat_v)[0]
        if idx.size > max_pixels_per_chip:
            idx = rng.choice(idx, max_pixels_per_chip, replace=False)
        X_list.append(flat_x[idx])
        y_list.append((flat_y[idx] == 1).astype("int8"))
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


def train_rf_baseline(
    meta: pd.DataFrame, split: Dict[str, List[str]], cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Treina RF no treino e avalia no teste (mesmo split espacial da CNN).

    Returns:
        Dicionário com métricas de teste + probabilidades para comparação.
    """
    if not _SK:
        raise RuntimeError("scikit-learn não está instalado.")
    from .metrics import SegMeter

    ignore_val = cfg["chips"]["label_ignore_value"]
    seed = cfg["project"]["random_seed"]

    logger.info("Baseline RF: carregando pixels de treino/teste...")
    Xtr, ytr = _load_pixels(meta, split["train"], ignore_val, seed=seed,
                            desc="RF pixels (treino)")
    Xte, yte = _load_pixels(meta, split["test"], ignore_val, seed=seed,
                            desc="RF pixels (teste)")

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=-1,
        class_weight="balanced", random_state=seed,
    )
    logger.info("Baseline RF: treinando Random Forest (300 árvores)...")
    clf.fit(Xtr, ytr)
    prob_te = clf.predict_proba(Xte)[:, 1]

    meter = SegMeter(ignore_val)
    meter.update(prob_te.reshape(1, -1), yte.reshape(1, -1), threshold=0.5)
    metrics = meter.compute()
    logger.info("Baseline RF | teste: %s", metrics)
    return {"metrics": metrics, "feature_importances": clf.feature_importances_.tolist()}
