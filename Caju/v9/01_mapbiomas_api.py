"""01 (v9) — MapBiomas Cobertura 10 m por município via API da Plataforma MapBiomas (SEM Earth Engine).

API: https://plataforma.brasil.mapbiomas.org/api/docs  (header tenant-id: mapbiomas ; projeto: brazil ; sem login)
Fluxo por município: busca território (geocode IBGE de 7 dígitos) -> POST /maps/exports
(subtema coverage_coverage_10m, ano) -> espera READY (QUEUED/EXPORTING/MOSAICKING/CLIPPING) -> baixa o GeoTIFF.
O limite de exportações simultâneas do projeto (HTTP 422) é tratado com espera e novas tentativas. Retomável.
Saída: data/mapbiomas_api/<regiao>_<ano>.tif  (reprojeção p/ a grade dos embeddings é feita no 02_chips_v9).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent
API = "https://plataforma.brasil.mapbiomas.org/api/v1/brazil"
H = {"tenant-id": "mapbiomas", "Content-Type": "application/json"}
SUB = "coverage_coverage_10m"
YEARS = [2025, 2024]          # usa 2025 se existir; senão 2024
OUT = ROOT / "data" / "mapbiomas_api"
LOG = ROOT / "logs" / "mapbiomas_api.log"


def log(msg):
    line = f"{time.strftime('%F %T')} | {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def territory_id(nome: str, geocode: str) -> str | None:
    r = requests.get(f"{API}/territories/search/{requests.utils.quote(nome)}", headers=H, timeout=60)
    r.raise_for_status()
    ts = [t for t in r.json().get("territories", []) if str(t.get("geocode")) == geocode]
    ts = [t for t in ts if t.get("categoryId") == 193]  # 193 = município (192 é outro território)
    return ts[0]["id"] if ts else None


def request_export(tid: str, year: int, max_wait_min=90):
    t0 = time.time()
    while time.time() - t0 < max_wait_min * 60:
        r = requests.post(f"{API}/maps/exports", headers=H, timeout=120,
                          json={"territoryId": tid, "subthemeKey": SUB, "years": [year],
                                "exportType": "range"})
        if r.status_code in (200, 201, 409):
            ex = r.json().get("exports") or [r.json()]
            return ex[0]["id"]
        if r.status_code == 422 and "limit" in r.text:
            time.sleep(60)  # limite de exportações simultâneas do projeto
            continue
        return ("ERRO", r.status_code, r.text[:200])
    return ("ERRO", "timeout fila", "")


def wait_ready(eid: str, max_min=60):
    t0 = time.time()
    while time.time() - t0 < max_min * 60:
        s = requests.get(f"{API}/maps/exports/{eid}", headers=H, timeout=60).json()
        st = s.get("status")
        if st == "READY":
            return s
        if st in ("FAILED", "EXPIRED"):
            return s
        time.sleep(20)
    return {"status": "TIMEOUT"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    import geopandas as gpd
    m = gpd.read_file(ROOT.parent / "dados_externos" / "municipios25" / "municipios25.gpkg")
    regs = [r for r in (ROOT.parent / "v8" / "data" / "aoi" / "regions.txt").read_text().split() if r.startswith("m")]
    for rid in regs:
        cod = rid[1:8]
        if list(OUT.glob(f"{rid}_*.tif")):
            log(f"MB pulado {rid} (já existe)")
            continue
        nome = str(m.loc[m["cod"].astype(str) == cod, "NM_MUN"].iloc[0])
        tid = territory_id(nome, cod)
        if not tid:
            log(f"MB ERRO {rid}: território não encontrado ({nome})")
            continue
        ok = False
        for y in YEARS:
            eid = request_export(tid, y)
            if isinstance(eid, tuple):
                log(f"MB {rid} {y}: pedido recusado {eid[1:]}")
                continue
            s = wait_ready(eid)
            if s.get("status") != "READY":
                log(f"MB {rid} {y}: status {s.get('status')} {str(s.get('errorMessage'))[:120]}")
                continue
            url = s.get("downloadUrl") or f"{API}/maps/exports/{eid}/download?kind=tif&tenantId=mapbiomas"
            r = requests.get(url, headers={"tenant-id": "mapbiomas"}, timeout=600, allow_redirects=True)
            r.raise_for_status()
            p = OUT / f"{rid}_{y}.tif"
            p.write_bytes(r.content)
            log(f"MB OK {rid} {y} | {len(r.content) / 1e6:.1f} MB | territorio {tid}")
            ok = True
            break
        if not ok:
            log(f"MB FALHOU {rid}")
    log("MB_API_FIM")


if __name__ == "__main__":
    main()
