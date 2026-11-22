"""05 — Amostras-candidatas de MANGA por SIMILARIDADE às amostras do usuário (Google Earth Pro).

Fluxo (GPU, por tile de 512 px — não monta mosaico de 64 bandas na RAM):
 1. Lê os polígonos da pasta `MANGA` (KML salvo em amostras_usuario/ ou myplaces.kml).
 2. Extrai os pixels de embeddings (64-d, unitários) dentro deles = conjunto-modelo P.
 3. Similaridade de cada pixel = média dos k maiores cossenos com P (kNN, k=5).
 4. Limiar calibrado: T = p10 da similaridade CRUZADA entre polígonos-modelo (cada polígono
    contra os demais); referência negativa = p99 nas classes MapBiomas seguras de não-manga.
 5. Candidatos = manchas ≥ 1 ha com sim ≥ T, fora das classes seguras de não-manga.
 6. KML para conferência no Google Earth Pro, estratificado por faixa de similaridade:
    alta / média / limiar + controle negativo (classe 48 MapBiomas com sim baixa).
Adaptado de Mapa_COCO/scripts/05 (manga): + polígonos extra (Livramento), modelo limitado a N px/polígono,
filtro contrastivo com protótipos de caju e coco (sim_manga - sim_neg >= margem), GPU com fração máxima.
Saídas: dados/rotulos/candidatos/{candidatos_<lista>.gpkg, candidatos_manga_<lista>.kml},
        v1/reports/candidatos_resumo.md
Uso: python 05_candidatos_similaridade.py [lista]
"""
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.features import rasterize, shapes
from rasterio.warp import Resampling, reproject, transform_bounds
from scipy import ndimage
from shapely.geometry import Polygon, shape

from common import cfg, ver

c = cfg()
lista = sys.argv[1] if len(sys.argv) > 1 else "top15_caatinga"
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cuda":
    torch.cuda.set_per_process_memory_fraction(c["gpu_fracao_max"])  # GPU dividida com o treino do caju
K = 5
out_dir = ver(c, "data", "rotulos", "candidatos", "x").parent
out_dir.mkdir(parents=True, exist_ok=True)
R = c["rotulos"]
SEG = set(R["classes_seguras_nao_coco_mb"])


# ---------- 1. polígonos do usuário ----------
def _kml_root(p):
    p = Path(p)
    if p.suffix.lower() == ".kmz":
        with zipfile.ZipFile(p) as z:
            return ET.fromstring(z.read(next(n for n in z.namelist() if n.endswith(".kml"))))
    return ET.parse(p).getroot()


def _strip(e):
    for el in e.iter():
        el.tag = re.sub(r"\{.*\}", "", el.tag)
    return e


def ler_amostras():
    a = c["amostras_usuario"]
    fontes = sorted(Path(a["kml_dir"]).glob("*.km[lz]"))  # sem myplaces: os 258 polígonos extra bastam p/ rodar
    polys = []
    for f in fontes:
        root = _strip(_kml_root(f))
        pastas = [fo for fo in root.iter("Folder") if (fo.findtext("name") or "").strip().upper() == a["pasta_ge"].upper()]
        alvo = pastas or ([root] if f.parent == Path(a["kml_dir"]) else [])
        for fo in alvo:
            for pm in fo.iter("Placemark"):
                for pg in pm.iter("Polygon"):
                    txt = pg.find(".//outerBoundaryIs//coordinates").text.split()
                    xy = [tuple(map(float, t.split(",")[:2])) for t in txt]
                    polys.append({"nome": pm.findtext("name") or f"amostra_{len(polys)}",
                                  "geometry": Polygon(xy), "fonte": f.name})
    extra = c["amostras_usuario"].get("poligonos_extra")
    if extra:
        ge = gpd.read_file(ver(c, extra)).to_crs(4326)
        polys += [{"nome": f"livramento_{i}", "geometry": g, "fonte": "manga_poligonos.gpkg"} for i, g in enumerate(ge.geometry)]
    if not polys:
        sys.exit(f"Nenhum polígono na pasta '{a['pasta_ge']}'. Salve a pasta MANGA do Google Earth Pro "
                 f"como KML em {a['kml_dir']} (botão direito > Salvar lugar como).")
    return gpd.GeoDataFrame(polys, crs=4326)


# ---------- utilidades de tile ----------
def emb_tile(src):
    a = src.read()                                     # (64,H,W) uint8
    valid = a[0] != 0
    x = (a.astype(np.float32) - 128.0) / 127.0
    return x, valid


def mb_tile(src):
    mb = np.zeros((src.height, src.width), np.uint8)
    with rasterio.open(c["externos"]["mapbiomas_2025"]) as m:
        b = transform_bounds(src.crs, m.crs, *src.bounds, densify_pts=21)
        w = m.window(*b).round_offsets().round_lengths()
        arr = m.read(1, window=w, boundless=True, fill_value=0)
        reproject(arr, mb, src_transform=m.window_transform(w), src_crs=m.crs,
                  dst_transform=src.transform, dst_crs=src.crs, resampling=Resampling.nearest)
    return mb


def knn_sim(x, P):
    """x: (64,H,W) np; P: (M,64) torch unitário -> (H,W) média dos K maiores cossenos."""
    C, H, W = x.shape
    v = torch.from_numpy(x.reshape(C, -1).T).to(dev, torch.float16)
    v = torch.nn.functional.normalize(v.float(), dim=1).half()
    out = torch.empty(v.shape[0], device=dev)
    for i in range(0, v.shape[0], 16384):
        s = v[i:i + 16384] @ P.T
        out[i:i + 16384] = s.topk(min(K, P.shape[0]), dim=1).values.float().mean(1)
    return out.reshape(H, W).cpu().numpy()


regioes = [l.split() for l in ver(c, "data", "aoi", f"regions_{lista}.txt").read_text().split("\n") if l.strip()]
emb_root = ver(c, "data", "embeddings", "2025", "x").parent

# ---------- 2. pixels-modelo ----------
am = ler_amostras()
amostras_px, ids_px = [], []
for rid, crs in regioes:
    tiles = sorted((emb_root / rid).glob("*.tif"))
    a_r = am.to_crs(crs)
    for t in tiles:
        with rasterio.open(t) as s:
            sel = a_r[a_r.intersects(Polygon.from_bounds(*s.bounds))]
            if sel.empty:
                continue
            x, valid = emb_tile(s)
            for j, (idx, r) in enumerate(sel.iterrows()):
                msk = rasterize([(r.geometry, 1)], out_shape=(s.height, s.width), transform=s.transform) == 1
                msk &= valid
                if msk.any():
                    amostras_px.append(x[:, msk].T)
                    ids_px += [idx] * int(msk.sum())
if not amostras_px:
    sys.exit("Os polígonos não caem em nenhuma região exportada.")
P_np = np.concatenate(amostras_px)
ids_px = np.array(ids_px)
_rng, _keep = np.random.default_rng(0), []
for i in np.unique(ids_px):
    w = np.nonzero(ids_px == i)[0]
    _keep += list(_rng.choice(w, min(len(w), c["modelo_px_por_poligono"]), replace=False))
_keep = np.sort(np.array(_keep)); P_np, ids_px = P_np[_keep], ids_px[_keep]
print(f"Polígonos-modelo: {len(am)} ({am['fonte'].value_counts().to_dict()})", flush=True)


def proto(vetor, emb_glob, px_por_poligono=None, **_):
    """Pixels-protótipo negativos (caju/coco): polígonos -> pixels dos tiles que os cobrem (N px por polígono)."""
    import glob
    g = gpd.read_file(vetor) if not str(vetor).lower().endswith((".kmz", ".kml")) else None
    if g is None:
        root = _strip(_kml_root(vetor)); xs = []
        for pg in root.iter("Polygon"):
            xs.append(Polygon([tuple(map(float, t.split(",")[:2])) for t in pg.find(".//outerBoundaryIs//coordinates").text.split()]))
        g = gpd.GeoDataFrame(geometry=xs, crs=4326)
    g = g[g.geometry.notna() & ~g.geometry.is_empty].reset_index(drop=True)
    rng, out = np.random.default_rng(1), []
    for t in sorted(glob.glob(emb_glob)):
        with rasterio.open(t) as s:
            gg = g.to_crs(s.crs)
            sel = gg[gg.intersects(Polygon.from_bounds(*s.bounds))]
            if sel.empty:
                continue
            x, valid = emb_tile(s)
            for geom in sel.geometry:
                m = (rasterize([(geom, 1)], out_shape=(s.height, s.width), transform=s.transform) == 1) & valid
                v = x[:, m].T
                if len(v):
                    out.append(v[rng.choice(len(v), min(len(v), px_por_poligono or c["modelo_px_por_poligono"]), replace=False)])
    return np.concatenate(out) if out else np.zeros((0, 64), np.float32)


_neg = {k: proto(**v) for k, v in c["contraste"].items() if isinstance(v, dict)}
NP = torch.nn.functional.normalize(torch.from_numpy(np.concatenate(list(_neg.values()))).to(dev), dim=1).half()
MARG = c["contraste"]["margem"]
print("Protótipos negativos: " + ", ".join(f"{k}={len(v)} px" for k, v in _neg.items()), flush=True)
P = torch.nn.functional.normalize(torch.from_numpy(P_np).to(dev), dim=1).half()

# limiar por similaridade cruzada (cada polígono contra os outros)
cross = []
for i in np.unique(ids_px):
    a_, b_ = P[torch.from_numpy(ids_px == i).to(dev)], P[torch.from_numpy(ids_px != i).to(dev)]
    if len(b_):
        cross.append((a_ @ b_.T).topk(min(K, len(b_)), dim=1).values.float().mean(1).cpu().numpy())
cross = np.concatenate(cross) if cross else np.array([0.9])
T = float(np.percentile(cross, 10))
DELTA = 0.03                      # faixa "explorar": T-DELTA <= sim < T (aprendizado ativo)
print(f"T={T:.4f} | sim cruzada p50={np.median(cross):.4f} | pixels-modelo={len(P_np)}", flush=True)

# ---------- 3–5. similaridade + candidatos ----------
reg_cand, neg_sims, linhas = [], [], []
n_contra = [0, 0]
for rid, crs in regioes:
    tiles = sorted((emb_root / rid).glob("*.tif"))
    if not tiles:
        linhas.append(f"| {rid} | sem tiles | – | – |")
        continue
    reg_poly = gpd.read_file(ver(c, "data", "aoi", "regions", f"{rid}.geojson")).set_crs(4326, allow_override=True).to_crs(crs)
    sim_out = out_dir / f"{rid}_sim.tif"
    cache, cstat = out_dir / f"_cache_{rid}.gpkg", out_dir / f"_cache_{rid}.npz"   # retomável por município
    if cstat.exists():
        z = np.load(cstat); nc, ns = list(z["nc"]), [z["ns"]] if z["ns"].size else []
        feats = gpd.read_file(cache).to_dict("records") if cache.exists() else []
    else:
        nc, ns = [0, 0], []
        feats = []
        for t in tiles:
            with rasterio.open(t) as s:
                x, valid = emb_tile(s)
                if not valid.any():
                    continue
                sim = knn_sim(x, P)
                sim[~valid] = 0
                simn = knn_sim(x, NP)
                contra = (sim - simn) < MARG          # mais parecido com caju/coco do que com manga
                mb = mb_tile(s)
                seg = np.isin(mb, list(SEG))
                if seg.any():
                    ns.append(np.random.default_rng(0).choice(sim[seg & valid], min(5000, int((seg & valid).sum())), replace=False))
                cand0 = (sim >= T - DELTA) & valid & ~seg
                nc[0] += int((cand0 & contra).sum()); nc[1] += int(cand0.sum())
                cand = cand0 & ~contra
                cand = ndimage.binary_opening(cand, iterations=1)
                for geom, v in shapes(cand.astype(np.uint8), mask=cand, transform=s.transform):
                    g = shape(geom)
                    if g.area >= R["mancha_min_ha"] * 1e4:
                        rr, cc = np.nonzero(rasterize([(g, 1)], out_shape=sim.shape, transform=s.transform))
                        feats.append({"regiao": rid, "sim_media": float(sim[rr, cc].mean()),
                                      "mb_moda": int(np.bincount(mb[rr, cc]).argmax()), "geometry": g})
                # controle negativo: classe 48 com similaridade baixa
                ctl = (mb == 48) & valid & (sim < T - 0.10)
                lab, n = ndimage.label(ndimage.binary_opening(ctl, iterations=2))
                for k in np.random.default_rng(1).permutation(np.arange(1, n + 1))[:2]:
                    rr, cc = np.nonzero(lab == k)
                    if rr.size >= 100:
                        x0, y0 = s.transform * (cc.mean(), rr.mean())
                        feats.append({"regiao": rid, "sim_media": float(sim[rr, cc].mean()), "mb_moda": 48,
                                      "geometry": Polygon.from_bounds(x0 - 50, y0 - 50, x0 + 50, y0 + 50), "controle": True})
        if feats:
            gpd.GeoDataFrame(feats, crs=crs).assign(controle=lambda d: d.get("controle", False)).fillna({"controle": False}).to_file(cache, driver="GPKG")
        np.savez(cstat, nc=np.array(nc), ns=np.concatenate(ns) if ns else np.zeros(0))
    n_contra[0] += int(nc[0]); n_contra[1] += int(nc[1]); neg_sims += ns
    print(f"{rid}: {len(feats)} manchas", flush=True)
    if not feats:
        linhas.append(f"| {rid} | 0 | 0 | – |")
        continue
    g = gpd.GeoDataFrame(feats, crs=crs)
    g["controle"] = g["controle"].fillna(False).astype(bool) if "controle" in g else False
    g = g[g.intersects(reg_poly.union_all())]
    pos = g[~g["controle"].astype(bool)].copy()
    pos["faixa"] = "explorar"
    acima = pos["sim_media"] >= T
    if acima.sum() >= 3:   # terços por posto entre as manchas acima do limiar
        pos.loc[acima, "faixa"] = pd.qcut(pos.loc[acima, "sim_media"].rank(method="first"), 3,
                                          labels=["limiar", "media", "alta"]).astype(str)
    elif acima.any():
        pos.loc[acima, "faixa"] = "alta"
    n_por = c["rotulos"]["fotointerp"]["por_municipio"]
    quota = {"alta": int(n_por * 0.3), "media": int(n_por * 0.25), "limiar": int(n_por * 0.15),
             "explorar": int(n_por * 0.2)}
    sel = pd.concat([pos[pos["faixa"] == f].sample(min(q, int((pos["faixa"] == f).sum())), random_state=42)
                     for f, q in quota.items()])
    ctl = g[g["controle"].astype(bool)].sample(min(int(n_por * 0.1), int(g["controle"].astype(bool).sum())), random_state=42)
    ctl["faixa"] = "controle_neg"
    sel = pd.concat([sel, ctl])
    sel["area_ha"] = sel.area / 1e4
    reg_cand.append(sel.to_crs(4326))
    linhas.append(f"| {rid} | {int(acima.sum())} ({pos[acima].area.sum() / 1e4:,.0f} ha) + {int((~acima).sum())} explorar | {len(sel)} | "
                  f"{pos['mb_moda'].value_counts().head(3).to_dict()} |")

neg = np.concatenate(neg_sims) if neg_sims else np.array([np.nan])
todos = gpd.GeoDataFrame(pd.concat(reg_cand, ignore_index=True), crs=4326)
todos["id"] = [f"c{i:04d}" for i in range(len(todos))]
todos["classe"] = ""
todos.drop(columns=["controle"], errors="ignore").to_file(out_dir / f"candidatos_{lista}.gpkg", driver="GPKG")

# ---------- 6. KML para o Google Earth Pro ----------
cor = {"alta": "ff00ff00", "media": "ff00ffff", "limiar": "ff0080ff", "explorar": "ff0000ff",
       "controle_neg": "ffff00ff"}
kml = ['<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
       f"<name>MANGA_candidatos_{lista}</name>"]
kml += [f'<Style id="{f}"><LineStyle><color>{v}</color><width>2</width></LineStyle>'
        f"<PolyStyle><fill>0</fill></PolyStyle></Style>" for f, v in cor.items()]
for rid, gr in todos.groupby("regiao"):
    kml.append(f"<Folder><name>{rid}</name>")
    for _, r in gr.sort_values("sim_media", ascending=False).iterrows():
        geom = r.geometry if r.geometry.geom_type == "Polygon" else max(r.geometry.geoms, key=lambda p: p.area)
        co = " ".join(f"{x:.6f},{y:.6f},0" for x, y in geom.simplify(0.00003).exterior.coords)
        kml.append(f"<Placemark><name>{r['id']}_{r['faixa']}</name><styleUrl>#{r['faixa']}</styleUrl>"
                   f"<description>sim={r['sim_media']:.3f}; area={r['area_ha']:.1f} ha; MapBiomas={r['mb_moda']}. "
                   f"Mover para MANGA_ok, NAO_MANGA ou DUVIDA. Manga: copas redondas densas verde-escuras em linhas (~4-10 m). Não-manga parecido: uva (latada plana), coco (estrela), banana (verde-claro, textura grossa), caju (copa irregular, sequeiro).</description>"
                   f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{co}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>")
    kml.append("</Folder>")
kml.append("</Document></kml>")
(out_dir / f"candidatos_manga_{lista}.kml").write_text("\n".join(kml), encoding="utf-8")

rep = ver(c, "reports", f"candidatos_resumo_{lista}.md")
rep.write_text("\n".join([
    "# Candidatos de manga por similaridade", "",
    f"Amostras do usuário: {len(am)} polígonos, {len(P_np)} pixels. k-NN k={K}.",
    f"Limiar T (p10 da similaridade cruzada) = {T:.3f}; similaridade cruzada p50 = {np.median(cross):.3f}.",
    f"Filtro contrastivo (caju/coco, margem {MARG}): {n_contra[0]:,} de {n_contra[1]:,} px candidatos removidos "
    f"({100 * n_contra[0] / max(n_contra[1], 1):.1f}%).",
    f"Classes seguras de não-manga (MapBiomas): p50 = {np.nanmedian(neg):.3f}, p99 = {np.nanpercentile(neg, 99):.3f}.",
    "", "| Região | Manchas ≥ T (+ faixa explorar) | Selecionadas p/ conferência | Classe MapBiomas (moda) |", "|---|---|---|---|",
    *linhas, "", f"KML: `{out_dir / f'candidatos_manga_{lista}.kml'}`"]), encoding="utf-8")
print(f"Candidatos: {len(todos)} em {todos['regiao'].nunique()} regiões | faixas {todos['faixa'].value_counts().to_dict()}")
print(f"Contraste removeu {100 * n_contra[0] / max(n_contra[1], 1):.1f}% dos px | resumo: {rep}")
