"""31 (v10b) — Rótulos com máscara MapBiomas Col. 11, 2025 (30 m reamostrada a 10 m).

Classes "compatíveis com caju" C = {48 outras lavouras perenes, 41 outras lavouras temporárias, 15 pastagem,
21 mosaico de usos}: contêm 98,6% dos polígonos de caju de Serra do Mel (reports/mb_classes_caju.csv).
Regra (sobre os rótulos por ranking da v10a):
  - fora de C (e com dado MapBiomas)  -> não-caju (0), exceto polígonos de campo
  - dentro de C                       -> mantém o rótulo do ranking (caju no topo / não-caju com nota baixa /
                                         ignorado no meio). Dentro de C, a lavoura perene pode ser coqueiro, manga etc.:
                                         só vira caju pelo ranking, nunca só pela classe.
  - anel de 2 px em volta do caju     -> ignorado
Saída: data/rotulos/<rid>_rot.tif e reports/rotulos_mb.md
"""
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

ROOT = Path(__file__).parent
_s = importlib.util.spec_from_file_location("f10", ROOT.parent / "v10/01_fontes.py"); f = importlib.util.module_from_spec(_s); _s.loader.exec_module(f)
C = [48, 41, 15, 21]
OUT = ROOT / "data/rotulos"; OUT.mkdir(parents=True, exist_ok=True)
res = []
for rid in f.rids():
    with rasterio.open(ROOT.parent / f"v10a/data/rotulos/{rid}_rot.tif") as d:
        R, prof = d.read(1), d.profile
    MB = rasterio.open(ROOT / f"data/mapbiomas/{rid}_mb.tif").read(1)
    PL = f.polys(R.shape, prof["transform"], prof["crs"]) if rid == "m2413359_SerradoMel" else np.zeros(R.shape, bool)
    fora = (MB != 0) & ~np.isin(MB, C) & ~PL
    pos0 = R == 1
    out = R.copy(); out[fora] = 0; out[PL] = 1
    pos = out == 1
    ring = ndimage.binary_dilation(pos, iterations=2) & ~pos
    out[ring & (out == 0) & ~fora] = 255
    with rasterio.open(OUT / f"{rid}_rot.tif", "w", **prof) as d:
        d.write(out, 1)
    res.append({"regiao": rid, "caju_v10a_ha": pos0.sum() / 100, "caju_v10b_ha": pos.sum() / 100,
                "caju_removido_mb_ha": (pos0 & fora).sum() / 100, "nao_v10a_ha": (R == 0).sum() / 100,
                "nao_v10b_ha": (out == 0).sum() / 100, "ign_v10b_ha": ((out == 255) & (MB != 0)).sum() / 100})
    print(res[-1], flush=True)
(ROOT / "reports/rotulos_mb.md").write_text("# Rótulos v10b (ranking v10a + máscara MapBiomas)\n\n"
                                            + pd.DataFrame(res).round(1).to_markdown(index=False), encoding="utf-8")
print("ROTULOS_MB_OK")
