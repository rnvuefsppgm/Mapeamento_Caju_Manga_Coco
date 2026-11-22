"""02 — Mapa ÚNICO 2025 (EPSG:32724, 10 m): fundo MapBiomas col. 11 + caju (97), manga (98), coco (99), nos municípios
mapeados por pelo menos uma cultura; 255 = fora. Pixel com 2+ culturas: vence a maior margem normalizada
(p − limiar)/(1 − limiar) de cada modelo. Decisão e contagem de áreas na GPU (torch), por blocos.
Saídas: saidas/mapa/culturas_2025_mapbiomas_c11.tif (paleta embutida) | saidas/auxiliar/disputas_2025.tif
(bits 1 caju, 2 manga, 4 coco onde houve disputa) | saidas/tabelas/areas_municipio_classe.csv (cod, classe, area_ha)
Uso: python 02_mapa_unico.py [teste <cod_ibge>]   (teste: só a caixa do município, em saidas/teste/)
"""
import re
import sys
import time
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from rasterio.windows import transform as wtransform
from shapely.geometry import box

from common import cfg, municipios_por_cultura, p

c = cfg()
dev = "cuda" if torch.cuda.is_available() else "cpu"
CRS, RES, B = c["crs"], c["res"], c["bloco"]
teste = len(sys.argv) > 2 and sys.argv[1] == "teste"
pref = ("saidas", "teste") if teste else ("saidas",)
f_map = p(*pref, "mapa", "culturas_2025_mapbiomas_c11.tif")
f_dis = p(*pref, "auxiliar", "disputas_2025.tif")
f_tab = p(*pref, "tabelas", "areas_municipio_classe.csv")
if f_tab.exists():
    print(f"já existe: {f_tab}"); sys.exit()

# --- municípios (id 1..N) e culturas mapeadas em cada um
mpc = municipios_por_cultura(c)
cods = sorted(set().union(*mpc.values()))
mun = gpd.read_file(c["municipios_shp"])
colc = next(k for k in mun.columns if k.upper().startswith("CD_MUN"))
coln = next(k for k in mun.columns if k.upper().startswith("NM_MUN"))
mun = mun[mun[colc].astype(int).isin(cods)].to_crs(CRS).reset_index(drop=True)
mun["cod"] = mun[colc].astype(int)
mun["mid"] = np.arange(1, len(mun) + 1)
CULT = list(c["culturas"])
NM = len(mun) + 1
cobre = np.zeros((NM, len(CULT)), bool)                        # cultura mapeada no município?
thr = np.full((NM, len(CULT)), 2.0, np.float32)               # limiar por município (2 = não mapeada)
for j, k in enumerate(CULT):
    v = c["culturas"][k]
    for r in mun.itertuples():
        if r.cod in mpc[k]:
            cobre[r.mid, j] = True
            thr[r.mid, j] = next((float(t) for t, l in v.get("limiar_por_municipio", {}).items() if r.cod in l), v["limiar"])
thr_t = torch.from_numpy(thr).to(dev)
codigo = torch.tensor([c["culturas"][k]["codigo"] for k in CULT], device=dev, dtype=torch.uint8)

# --- grade de saída
alvo = mun[mun.cod == int(sys.argv[2])] if teste else mun
x0, y0, x1, y1 = alvo.total_bounds
x0, y0 = np.floor(x0 / RES) * RES, np.floor(y0 / RES) * RES
x1, y1 = np.ceil(x1 / RES) * RES, np.ceil(y1 / RES) * RES
W, H = int((x1 - x0) / RES), int((y1 - y0) / RES)
T = from_origin(x0, y1, RES, RES)
print(f"grade {W}x{H} px ({W * H / 1e9:.2f} Gpx), {len(mun)} municípios, {dev}", flush=True)


def vrt(path):
    s = rasterio.open(path)
    return WarpedVRT(s, crs=CRS, transform=T, width=W, height=H, resampling=Resampling.nearest, nodata=255 if s.nodata is None and "coverage" not in str(path) else s.nodata)


src = {k: (vrt(v["classe"]), vrt(v["prob"])) for k, v in c["culturas"].items()}
mb = WarpedVRT(rasterio.open(c["mapbiomas"]), crs=CRS, transform=T, width=W, height=H, resampling=Resampling.nearest, nodata=0)


# --- paleta: cores MapBiomas da planilha da legenda + 3 culturas
def paleta():
    cm = {i: (200, 200, 200, 255) for i in range(256)}
    cm[255] = (0, 0, 0, 0)
    try:
        z = zipfile.ZipFile(c["legenda_mapbiomas"])
        ss = [re.sub("<[^>]+>", "", s) for s in re.findall(r"<si>(.*?)</si>", z.read("xl/sharedStrings.xml").decode("utf8"), re.S)]
        for row in re.findall(r"<row[^>]*>(.*?)</row>", z.read("xl/worksheets/sheet1.xml").decode("utf8"), re.S):
            vals = []
            for attr, inner in re.findall(r"<c r=\"[A-Z]+\d+\"([^>]*?)>(.*?)</c>", row, re.S):
                m = re.search(r"<v>(.*?)</v>", inner)
                if m:
                    vals.append(ss[int(m.group(1))] if 't="s"' in attr else m.group(1))
            cor = next((v for v in vals if re.fullmatch(r"#[0-9a-fA-F]{6}", v.strip())), None)
            num = [float(v) for v in vals if re.fullmatch(r"\d+(\.0)?", v.strip())]
            if cor and num and 0 < num[-1] < 100:
                h = cor.strip().lstrip("#")
                cm[int(num[-1])] = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (255,)
    except Exception as e:  # paleta é cosmética: segue sem ela
        print(f"aviso paleta: {e}")
    for v in c["culturas"].values():
        h = v["cor"].lstrip("#")
        cm[v["codigo"]] = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (255,)
    return cm


prof = dict(driver="GTiff", height=H, width=W, count=1, dtype="uint8", crs=CRS, transform=T, nodata=255,
            tiled=True, blockxsize=512, blockysize=512, compress="deflate", BIGTIFF="YES")
cont = torch.zeros(NM * 256, dtype=torch.int64, device=dev)     # contagem (município, classe)
disp = torch.zeros(NM * 8, dtype=torch.int64, device=dev)       # contagem (município, padrão de disputa)
sidx = mun.sindex
t0 = time.time(); nb = 0; nfeitos = 0
with rasterio.open(f_map, "w", **prof) as om, rasterio.open(f_dis, "w", **prof) as od:
    om.write_colormap(1, paleta())
    for r0 in range(0, H, B):
        for c0 in range(0, W, B):
            nb += 1
            win = Window(c0, r0, min(B, W - c0), min(B, H - r0))
            wt = wtransform(win, T)
            bx = box(*rasterio.windows.bounds(win, T))
            ii = sidx.query(bx, predicate="intersects")
            if not len(ii):
                continue
            mid = rasterize([(g, int(m)) for g, m in zip(mun.geometry.iloc[ii], mun.mid.iloc[ii])],
                            out_shape=(win.height, win.width), transform=wt, fill=0, dtype="int32")
            if not mid.any():
                continue
            nfeitos += 1
            m_t = torch.from_numpy(mid).to(dev).long()
            fora = m_t == 0
            out = torch.from_numpy(mb.read(1, window=win)).to(dev)
            score = torch.full((len(CULT),) + out.shape, -1e9, device=dev)
            flag = torch.zeros((len(CULT),) + out.shape, dtype=torch.bool, device=dev)
            for j, k in enumerate(CULT):
                cl = torch.from_numpy(src[k][0].read(1, window=win)).to(dev)
                pr = torch.from_numpy(src[k][1].read(1, window=win)).to(dev).float().clamp_(max=100) / 100
                t = thr_t[m_t, j]
                flag[j] = (cl == 1) & (t < 1.5)                 # só onde a cultura foi mapeada no município
                score[j] = torch.where(flag[j], (pr - t) / (1 - t).clamp(min=1e-3), score[j])
            nf = flag.sum(0)
            venc = score.argmax(0)
            out = torch.where(nf > 0, codigo[venc], out)
            out[fora] = 255
            bits = (flag[0].byte() * 1 + flag[1].byte() * 2 + flag[2].byte() * 4) if len(CULT) == 3 else nf.byte()
            dis = torch.where(nf > 1, bits, torch.zeros_like(bits))
            dis[fora] = 255
            cont += torch.bincount((m_t * 256 + out.long()).ravel(), minlength=NM * 256)
            disp += torch.bincount((m_t * 8 + torch.where(fora, 0, dis).long()).ravel(), minlength=NM * 8)
            om.write(out.cpu().numpy(), 1, window=win)
            od.write(dis.cpu().numpy(), 1, window=win)
            if nfeitos % 20 == 0:
                print(f"bloco {nb}: {nfeitos} com município, {time.time() - t0:.0f}s", flush=True)
    print(f"blocos {nb} ({nfeitos} com município) em {time.time() - t0:.0f}s; overviews...", flush=True)
for f in (f_map, f_dis):
    with rasterio.open(f, "r+") as d:
        d.build_overviews([2, 4, 8, 16, 32, 64], Resampling.nearest)
        d.update_tags(ns="rio_overview", resampling="nearest")

# --- tabela longa (município × classe) e disputas
A = cont.view(NM, 256).cpu().numpy()[1:] / 100.0       # ha (pixel 10 m = 0,01 ha)
D = disp.view(NM, 8).cpu().numpy()[1:] / 100.0
L = [dict(cod=r.cod, municipio=getattr(r, coln), classe=cl, area_ha=A[r.mid - 1, cl]) for r in mun.itertuples() for cl in np.nonzero(A[r.mid - 1])[0]]
L += [dict(cod=r.cod, municipio=getattr(r, coln), classe=f"disputa_{b}", area_ha=D[r.mid - 1, b]) for r in mun.itertuples() for b in (3, 5, 6, 7) if D[r.mid - 1, b]]
pd.DataFrame(L).to_csv(f_tab, index=False)
cobt = pd.DataFrame(cobre[1:], columns=[f"mapeado_{k}" for k in CULT]); cobt.insert(0, "cod", mun.cod.values)
cobt.to_csv(p(*pref, "tabelas", "cobertura_municipios.csv"), index=False)
for k in CULT:
    cd = c["culturas"][k]["codigo"]
    print(f"{k} ({cd}): {A[:, cd].sum():,.0f} ha")
print(f"disputas: {D[:, 1:].sum():,.0f} ha | FIM {time.time() - t0:.0f}s -> {f_map}")
