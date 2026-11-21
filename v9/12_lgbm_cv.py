"""12 (v9) — Candidato C4: LightGBM por pixel (embeddings + média/desvio 3x3) nos MESMOS 5 folds.

Treino: amostra de pixels dos chips (campo + pseudo variante A), balanceada; nunca blocos de val/teste.
Teste: todos os pixels dos chips de campo dos blocos de teste. Saída: reports/oof/lgbm_A_foldN.npz.
Leve na RAM (amostragem) e roda em CPU (4 threads, para não disputar com a GPU).
"""
import importlib.util
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

from src.config import load_config, resolve_path
from src.dataset import load_chip, pseudo_ids_for_train

ROOT = Path(__file__).parent
cfg = load_config(None)
spec = importlib.util.spec_from_file_location("cv", ROOT / "08_spatial_cv.py")
cv = importlib.util.module_from_spec(spec); spec.loader.exec_module(cv)
meta = pd.read_parquet(resolve_path(cfg, "metadata") / "chips_metadata_A.parquet")
field = meta[meta.source == "field"]
groups = sorted(field.block_id.dropna().unique().tolist())
folds = cv._make_folds(groups, 5, cfg["project"]["random_seed"])
rng = np.random.default_rng(42)


def feats(x):  # (64,H,W) -> (H*W, 192)
    mu = uniform_filter(x, size=(1, 3, 3))
    sd = np.sqrt(np.maximum(uniform_filter(x * x, size=(1, 3, 3)) - mu * mu, 0))
    return np.concatenate([x, mu, sd]).reshape(192, -1).T


def sample(ids, n_px=1500):
    X, Y = [], []
    for _, r in meta[meta.chip_id.isin(ids)].iterrows():
        x, y = load_chip(r, cfg)
        F, yy = feats(x), y.ravel()
        ok = np.where(yy != 255)[0]
        if ok.size == 0:
            continue
        pos, neg = ok[yy[ok] == 1], ok[yy[ok] == 0]
        k = n_px // 2
        sel = np.concatenate([rng.choice(pos, min(k, pos.size), replace=False) if pos.size else [],
                              rng.choice(neg, min(k, neg.size), replace=False) if neg.size else []]).astype(int)
        X.append(F[sel]); Y.append(yy[sel])
    return np.concatenate(X).astype("float32"), np.concatenate(Y)


od = ROOT / "reports" / "oof"
for i, test_b in enumerate(folds):
    rem = [g for g in groups if g not in set(test_b)]
    val_b = [rem[i % len(rem)]]
    tr_ids = field[field.block_id.isin([g for g in rem if g not in val_b])].chip_id.tolist()
    tr_ids += pseudo_ids_for_train(meta, set(val_b) | set(test_b))
    X, Y = sample(tr_ids)
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
                             colsample_bytree=0.5, n_jobs=4, verbose=-1)
    clf.fit(X, Y)
    P, T = [], []
    for _, r in field[field.block_id.isin(test_b)].iterrows():
        x, y = load_chip(r, cfg)
        P.append(clf.predict_proba(feats(x))[:, 1].reshape(y.shape).astype("float16")); T.append(y.astype("uint8"))
    np.savez_compressed(od / f"lgbm_A_fold{i + 1}.npz", p=np.stack(P), t=np.stack(T))
    print(f"fold {i + 1}: treino {len(Y)} px, teste {len(T)} chips", flush=True)
