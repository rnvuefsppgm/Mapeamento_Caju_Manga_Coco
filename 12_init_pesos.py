"""12 (v1) — Pesos iniciais para o teste A/B (proteção 3):
A = caju v10b (final_seed42) com a CABEÇA reiniciada (proteção 1): aproveita encoder/decoder, esquece "caju".
B = ImageNet (encoder genérico, resto aleatório).
Saídas: modelos/init/{A_caju_sem_cabeca,B_imagenet}.pt no formato lido por src/ft.load_init (FT_INIT).
"""
from pathlib import Path

import torch

from src.config import load_config, resolve_path
from src.models import build_model

cfg = load_config(Path(__file__).parent / "config/config.yaml")
od = resolve_path(cfg, "models") / "init"; od.mkdir(parents=True, exist_ok=True)
torch.manual_seed(42)
novo = build_model(cfg, in_channels=64).state_dict()
ck = torch.load(cfg["treino_v1"]["init_caju"], map_location="cpu", weights_only=False)["model_state"]
cab = [k for k in novo if k.startswith("segmentation_head.")]
a = {k: (novo[k] if k in cab else ck[k]) for k in novo}
torch.save({"model_state": a, "test_blocks": "-", "init": "caju v10b seed42, cabeça reiniciada"}, od / "A_caju_sem_cabeca.pt")
torch.save({"model_state": novo, "test_blocks": "-", "init": "ImageNet"}, od / "B_imagenet.pt")
print(f"init A (caju, {len(cab)} tensores da cabeça reiniciados) e B (ImageNet) salvos em {od}")
