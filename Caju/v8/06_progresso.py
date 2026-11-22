"""06 (v8) — Tabela de progresso da predição: área mapeada × PAM, horário e duração por município.

Duração = diferença entre o término do município e o anterior (a predição é sequencial).
Motor: CPU (04) até a troca às 19:15:06; GPU (04b) depois. Saída: reports/progresso_predicao.md.
Uso: python 06_progresso.py [--ultimos N]  (imprime a tabela; com --ultimos só as N últimas linhas)
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
TROCA = datetime(2026, 11, 20, 19, 15, 6)
r = pd.DataFrame(json.loads((ROOT / "reports" / "area_caju_2025.json").read_text(encoding="utf-8")))
r = r[r.regiao.str.startswith("m")].copy()
r["fim"] = [datetime.fromtimestamp(os.path.getmtime(ROOT / "predictions" / "2025" / f"{x}_prob.tif"))
            for x in r.regiao]
r = r.sort_values("fim").reset_index(drop=True)
ini = datetime(2026, 11, 20, 18, 10, 13)  # início da predição (log)
prev = [ini] + list(r.fim[:-1])
r["dur_s"] = [(f - max(p, TROCA) if f > TROCA >= p else f - p).total_seconds() for f, p in zip(r.fim, prev)]
r["motor"] = ["GPU" if f > TROCA else "CPU" for f in r.fim]
r["cod"] = r.regiao.str[1:8].astype(int)
pam = pd.read_csv(ROOT.parent / "dados_externos" / "ranking_caju_caatinga_2023_2025.csv")
r = r.merge(pam[["cod", "nome", "area_ha_2025"]], on="cod", how="left")
r["mapa/PAM"] = (r.caju_ha / r.area_ha_2025).round(2)
fmt = lambda s: f"{int(s // 60)} min {int(s % 60):02d} s" if s >= 60 else f"{int(s)} s"
t = pd.DataFrame({"#": range(1, len(r) + 1), "Município": r.nome, "Motor": r.motor,
                  "Término": r.fim.dt.strftime("%H:%M:%S"), "Duração": r.dur_s.map(fmt),
                  "Mapa (ha)": r.caju_ha.map(lambda v: f"{v:,.0f}".replace(",", ".")),
                  "PAM 2025 (ha)": r.area_ha_2025.map(lambda v: f"{v:,.0f}".replace(",", ".")),
                  "Mapa/PAM": r["mapa/PAM"]})
md = t.to_markdown(index=False, disable_numparse=True)
(ROOT / "reports" / "progresso_predicao.md").write_text(
    f"# Progresso da predição 2025 (v8)\n\n{len(t)} de 25 municípios\n\n{md}\n", encoding="utf-8")
n = int(sys.argv[sys.argv.index("--ultimos") + 1]) if "--ultimos" in sys.argv else None
print(t.tail(n).to_markdown(index=False, disable_numparse=True) if n else md)
