"""15 (v10) — Predição final 2025 dos 25 municípios com o modelo final (3 sementes) na GPU.
Reusa o motor do v8/04b (blocos, janelas na GPU, recorte ao município); saída em v10/predictions/2025/.
"""
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent
V8 = ROOT.parent / "v8"
_s = importlib.util.spec_from_file_location("g", V8 / "04b_predict_emb_gpu.py")
g = importlib.util.module_from_spec(_s); _s.loader.exec_module(g)
from src.config import load_config  # noqa: E402  (src da v7, inserido no path pelo 04b)
from src.models import build_model  # noqa: E402

cfg = load_config(str(ROOT.parent / "v7/config/config.yaml")); cfg["experiment"]["channels"] = "emb"
dev = g.get_device()
mem = []
for f in sorted((ROOT / "models").glob("final_seed*.pt")):
    ck = torch.load(f, map_location="cpu", weights_only=False)
    m = build_model(cfg, in_channels=64); m.load_state_dict(ck["model_state"])
    mem.append((m.to(dev).eval(), ck["stats"], float(ck["threshold"])))
g.ROOT = ROOT  # saídas em v10/predictions/2025 (o recorte usa ROOT.parent/dados_externos)
(ROOT / "predictions/2025").mkdir(parents=True, exist_ok=True)
res = []
for v in sorted((V8 / "data/ee_exports/2025/embeddings").glob("m*.vrt")):
    r = g.predict_region(v, mem, cfg, dev)
    res.append(r); print(r, flush=True)
    (ROOT / "reports/area_caju_2025_v10.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"PREDICT_V10_OK {len(res)} municipios", file=sys.stderr)
