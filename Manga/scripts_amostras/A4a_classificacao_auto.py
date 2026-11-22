"""A4a — Classificação AUTOMÁTICA dos candidatos da rodada 1 em MANGA_ok / NAO_MANGA / DUVIDA por probabilidade.

Método (detalhe em docs/A4a_CLASSIFICACAO_AUTO.md):
 1. XGBoost multiclasse na GPU sobre os 64 embeddings 2025: 0 outros | 1 manga | 2 caju | 3 coco.
    manga = polígonos de Livramento (código 1 do raster de campo); outros = Livramento código 3 (área amostrada,
    > 100 m dos pomares) + classes seguras MapBiomas nos 15 municípios; caju = Pol_Caju (Serra do Mel); coco = COCO.kmz.
 2. Calibração: validação cruzada espacial (blocos de 3 km em Livramento; demais por sorteio) -> p_manga fora do fold
    -> regressão isotônica (p calibrada). Relatório com AUC, Brier e tabela de confiabilidade.
 3. Candidato: média de p (calibrada) nos seus pixels; regras de config `amostragem.auto_classificacao`.
Saídas: dados/amostras_usuario/rodada1_auto.kmz (3 pastas) + rodada1_checagem.kml (amostra p/ conferência),
        dados/rotulos/candidatos/rodada1_auto.gpkg, reports/A4a_classificacao_auto.md. Imprime <= 20 linhas.
Rótulos daqui são AUTOMÁTICOS (origem="auto"): servem para treino/modelo da rodada 2, NUNCA para avaliação.
"""
import glob
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xgboost as xgb
import yaml
from rasterio.features import rasterize, shapes
from scipy import ndimage
from shapely.geometry import shape
from rasterio.windows import Window, from_bounds
from shapely.geometry import Polygon
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

RAIZ = Path(__file__).resolve().parents[1]
CF = yaml.safe_load(open(RAIZ / "config/config.yaml", encoding="utf-8"))
A, AC = CF["amostragem"], CF["amostragem"]["auto_classificacao"]
TR = AC["treino"]
D = RAIZ / "dados"
lista = next((a for a in sys.argv[1:] if not a.startswith("--")), "top15_caatinga")
rng = np.random.default_rng(42)
SEG = A["rotulos"]["classes_seguras_nao_coco_mb"]
regioes = [l.split()[0] for l in (D / "aoi" / f"regions_{lista}.txt").read_text().split("\n") if l.strip()]
LIV = next(r for r in regioes if r.startswith("m2919504"))


def dec(a):  # uint8 cru -> float (x*127+128); NoData 0
    return (a.astype(np.float32) - 128.0) / 127.0


def janela(src, bounds):
    w = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
    return w.intersection(Window(0, 0, src.width, src.height))


# ---------------- 1. dados de treino ----------------
X, y, grp = [], [], []
with rasterio.open(D / "embeddings" / f"{LIV}.vrt") as e, rasterio.open(D / "rotulos" / f"{LIV}_campo.tif") as k:
    K = k.read(1)
    rr, cc = np.nonzero(K == 1)
    w = Window(cc.min() - 300, rr.min() - 300, cc.max() - cc.min() + 600, rr.max() - rr.min() + 600)
    w = w.intersection(Window(0, 0, e.width, e.height))
    E, Kw = e.read(window=w), K[w.row_off:w.row_off + w.height, w.col_off:w.col_off + w.width]
    tw = e.window_transform(w)
    for cod, cl, nmax in [(1, 1, TR["pos_max_px"]), (3, 0, TR["neg_denso_px"])]:
        r, c = np.nonzero((Kw == cod) & (E[0] != 0))
        s = rng.choice(len(r), min(nmax, len(r)), replace=False)
        r, c = r[s], c[s]
        xs, ys = tw * (c + 0.5, r + 0.5)
        X.append(dec(E[:, r, c]).T); y += [cl] * len(r)
        grp += [f"liv_{int(a // 3000)}_{int(b // 3000)}" for a, b in zip(xs, ys)]   # blocos de 3 km
    del E
for rid in regioes:   # classes seguras MapBiomas: até 6 janelas de 512 px por município
    with rasterio.open(D / "embeddings" / f"{rid}.vrt") as e, rasterio.open(D / "mapbiomas" / f"{rid}_mb.tif") as m:
        got = 0
        for _ in range(30):
            if got >= TR["neg_mb_px_por_regiao"]:
                break
            w = Window(int(rng.integers(0, max(1, e.width - 512))), int(rng.integers(0, max(1, e.height - 512))), 512, 512)
            M = m.read(1, window=w)
            ok = np.isin(M, SEG)
            if ok.sum() < 200:
                continue
            E = e.read(window=w)
            r, c = np.nonzero(ok & (E[0] != 0))
            s = rng.choice(len(r), min(TR["neg_mb_px_por_regiao"] // 6 + 1, len(r)), replace=False)
            X.append(dec(E[:, r[s], c[s]]).T); y += [0] * len(s); grp += [f"mb_{rid}"] * len(s)
            got += len(s)


def px_poligonos(g, emb_glob, n, cl, nome):
    out = 0
    for t in sorted(glob.glob(emb_glob)):
        with rasterio.open(t) as s:
            gg = g.to_crs(s.crs)
            sel = gg[gg.intersects(Polygon.from_bounds(*s.bounds))]
            if sel.empty:
                continue
            E = s.read()
            for geom in sel.geometry:
                msk = (rasterize([(geom, 1)], out_shape=(s.height, s.width), transform=s.transform) == 1) & (E[0] != 0)
                r, c = np.nonzero(msk)
                if len(r):
                    k = rng.choice(len(r), min(n, len(r)), replace=False)
                    X.append(dec(E[:, r[k], c[k]]).T); y.extend([cl] * len(k)); grp.extend([nome] * len(k))
                    out += len(k)
    return out


def ler_kml_poligonos(p):
    import re
    z = zipfile.ZipFile(p)
    t = z.read(next(n for n in z.namelist() if n.endswith(".kml"))).decode("utf8", "ignore")
    pols = [Polygon([tuple(map(float, q.split(",")[:2])) for q in s.split()])
            for s in re.findall(r"<outerBoundaryIs>.*?<coordinates>(.*?)</coordinates>", t, re.S)]
    return gpd.GeoDataFrame(geometry=pols, crs=4326)


ct = A["contraste"]
n_caju = px_poligonos(gpd.read_file(ct["caju"]["vetor"]).dropna(subset=["geometry"]), ct["caju"]["emb_glob"],
                      TR["caju_px_por_poligono"], 2, "caju")
n_coco = px_poligonos(ler_kml_poligonos(ct["coco"]["vetor"]), ct["coco"]["emb_glob"], TR["coco_px_por_poligono"], 3, "coco")
X, y, grp = np.concatenate(X), np.array(y), np.array(grp)
print(f"Treino: {len(y):,} px | manga {(y == 1).sum():,} · outros {(y == 0).sum():,} · caju {n_caju:,} · coco {n_coco:,}", flush=True)

# ---------------- 2. CV espacial + calibração ----------------
ug = np.unique(grp)
fold_of = dict(zip(ug, rng.integers(0, TR["folds"], len(ug))))
fold = np.array([fold_of[g] for g in grp])
um = np.isin(grp, ["caju", "coco"])            # classe com um só grupo: folds por pixel (todas as classes em todo treino)
fold[um] = rng.integers(0, TR["folds"], int(um.sum()))
cnt = np.bincount(y, minlength=4)
wcls = (len(y) / (4 * np.maximum(cnt, 1)))
par = dict(objective="multi:softprob", num_class=4, tree_method="hist", device="cuda", **TR["xgb"])
oof = np.zeros((len(y), 4), np.float32)
for f in range(TR["folds"]):
    tr, te = fold != f, fold == f
    m = xgb.XGBClassifier(**par).fit(X[tr], y[tr], sample_weight=wcls[y[tr]])
    oof[te] = m.predict_proba(X[te])
iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof[:, 1], (y == 1).astype(float))
pc = iso.predict(oof[:, 1])
auc, brier = roc_auc_score(y == 1, oof[:, 1]), brier_score_loss(y == 1, pc)
bins = pd.cut(pc, [0, .2, .5, .8, .9, 1.0001], include_lowest=True)
conf = pd.DataFrame({"bin": bins, "manga": y == 1}).groupby("bin", observed=False).agg(n=("manga", "size"), frac_manga=("manga", "mean"))
modelo = xgb.XGBClassifier(**par).fit(X, y, sample_weight=wcls[y])
print(f"CV espacial: AUC manga {auc:.3f} | Brier (calibrado) {brier:.3f}", flush=True)
MD = RAIZ / "modelos" / "a4a"; MD.mkdir(parents=True, exist_ok=True)   # reuso no A5 (mapa por pixel)
modelo.save_model(MD / "xgb_4classes.json")
np.savez(MD / "isotonica.npz", x=iso.X_thresholds_, y=iso.y_thresholds_)
(MD / "cv.txt").write_text(f"AUC {auc:.4f} | Brier {brier:.4f} | treino {len(y)} px" + chr(10) + conf.to_string() + chr(10), encoding="utf-8")
if "--so-modelo" in sys.argv:
    sys.exit(0)
del X

# ---------------- 3. candidatos ----------------
cand = gpd.read_file(D / "rotulos" / "candidatos" / f"candidatos_{lista}.gpkg")
cand[["p_manga", "p_caju", "p_coco", "n_px", "area_conf_ha", "frac_conf"]] = np.nan
OKP = AC["ok_pixel"]
geom_ok = {}
for rid, sub in cand.groupby("regiao"):
    conhecida = int(rid[1:8]) in AC["regiao_conhecida"]
    with rasterio.open(D / "embeddings" / f"{rid}.vrt") as e:
        sg = sub.to_crs(e.crs)
        for idx, geom in zip(sg.index, sg.geometry):
            w = janela(e, geom.bounds)
            if w.width < 1 or w.height < 1:
                continue
            E = e.read(window=w)
            msk = (rasterize([(geom, 1)], out_shape=(int(w.height), int(w.width)), transform=e.window_transform(w)) == 1) & (E[0] != 0)
            if not msk.any():
                continue
            P = modelo.predict_proba(dec(E[:, msk]).T)
            pm = iso.predict(P[:, 1])
            # opção A: pixels confiáveis (p calibrada por pixel); região nova exige também margem sobre caju/coco
            cf = pm >= AC["p_ok"] if conhecida else (pm >= AC["p_ok_regiao_nova"]) &                 (pm - np.maximum(P[:, 2], P[:, 3]) >= AC["margem_caju_coco"])
            C = np.zeros(msk.shape, bool); C[msk] = cf
            C = ndimage.binary_opening(C, iterations=1)
            lab, n = ndimage.label(C)
            if n:
                tam = ndimage.sum(C, lab, range(1, n + 1))
                C = np.isin(lab, np.nonzero(tam >= OKP["componente_min_px"])[0] + 1)
            if C.any():
                partes = [shape(g) for g, v in shapes(C.astype(np.uint8), mask=C, transform=e.window_transform(w))]
                geom_ok[idx] = gpd.GeoSeries(partes, crs=e.crs).union_all()
            cand.loc[idx, ["p_manga", "p_caju", "p_coco", "n_px", "area_conf_ha", "frac_conf"]] = [
                float(pm.mean()), float(P[:, 2].mean()), float(P[:, 3].mean()), int(msk.sum()),
                C.sum() / 100.0, C.sum() / msk.sum()]

cand["cod"] = cand.regiao.str[1:8].astype(int)
cand["regiao_conhecida"] = cand.cod.isin(AC["regiao_conhecida"])
cand["margem"] = cand.p_manga - cand[["p_caju", "p_coco"]].max(axis=1)
compat = cand.mb_moda.isin(AC["classes_compativeis"])
ctl = cand.faixa == "controle_neg"
ok = ~ctl & compat & (cand.area_conf_ha >= OKP["area_min_ha"]) & (cand.frac_conf >= OKP["frac_min"])
nao = (cand.p_manga <= AC["p_nao"]) | (ctl & (cand.p_manga < AC["p_nao_controle"]))
cand["pasta"] = np.select([ok, nao], ["MANGA_ok", "NAO_MANGA"], "DUVIDA")
cand["motivo"] = ""
# teto PAM: área MANGA_ok <= teto x área colhida 2025; excedente (menor p) -> DUVIDA
pam = pd.read_csv(D / "pam" / "pam_manga_2023_2025.csv").set_index("cod")["area_colhida_ha_2025"]
for cod, sub in cand[cand.pasta == "MANGA_ok"].sort_values("p_manga", ascending=False).groupby("cod"):
    lim = AC["teto_pam"] * pam.get(cod, 0)
    exc = sub.index[sub.area_conf_ha.cumsum() > lim]
    cand.loc[exc, ["pasta", "motivo"]] = ["DUVIDA", "acima do teto PAM"]
cand["area_ha_candidato"] = cand.area_ha
for idx in cand.index[cand.pasta == "MANGA_ok"]:        # MANGA_ok = só o recorte confiável
    cand.loc[[idx], "geometry"] = gpd.GeoSeries([geom_ok[idx]], crs=CF["manga"]["crs"]).to_crs(4326).values
    cand.loc[idx, "area_ha"] = cand.loc[idx, "area_conf_ha"]
cand["origem"] = "auto"
cand.loc[cand.p_manga.isna(), ["pasta", "motivo"]] = ["DUVIDA", "sem pixel válido"]
cand.to_file(D / "rotulos" / "candidatos" / f"rodada1_auto.gpkg", driver="GPKG")

# ---------------- 4. KMZ (3 pastas) + KML de checagem ----------------
COR = {"MANGA_ok": "ff00ff00", "NAO_MANGA": "ff0000ff", "DUVIDA": "ff00ffff"}


def kml(df, nome, pastas):
    L = ['<?xml version="1.0" encoding="UTF-8"?>', '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>', f"<name>{nome}</name>"]
    L += [f'<Style id="{k}"><LineStyle><color>{v}</color><width>2</width></LineStyle><PolyStyle><fill>0</fill></PolyStyle></Style>'
          for k, v in COR.items()]
    for pasta in pastas:
        L.append(f"<Folder><name>{pasta}</name>")
        for r in df[df.pasta_kml == pasta].itertuples():
            gs = [r.geometry] if r.geometry.geom_type == "Polygon" else list(r.geometry.geoms)
            geo = "".join("<Polygon><outerBoundaryIs><LinearRing><coordinates>" + " ".join(f"{x:.7f},{yv:.7f}" for x, yv in g.exterior.coords)
                          + "</coordinates></LinearRing></outerBoundaryIs></Polygon>" for g in gs)
            L.append(f"<Placemark><name>{r.id}</name><styleUrl>#{r.pasta}</styleUrl><description>"
                     f"auto: {r.pasta} | p_manga {r.p_manga:.2f} · p_caju {r.p_caju:.2f} · p_coco {r.p_coco:.2f} | "
                     f"{r.regiao} | faixa {r.faixa} | MapBiomas {r.mb_moda} | {r.area_ha:.1f} ha (candidato {r.area_ha_candidato:.1f} ha; "
                     f"{100 * (r.frac_conf if r.frac_conf == r.frac_conf else 0):.0f}% confiável) {r.motivo}</description>"
                     f"<MultiGeometry>{geo}</MultiGeometry></Placemark>")
        L.append("</Folder>")
    return "\n".join(L + ["</Document></kml>"])


c2 = cand.dropna(subset=["p_manga"]).copy()
c2["pasta_kml"] = c2.pasta
kmz = D / "amostras_usuario" / "rodada1_auto.kmz"
with zipfile.ZipFile(kmz, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("doc.kml", kml(c2, "MANGA rodada 1 (classificação automática)", ["MANGA_ok", "NAO_MANGA", "DUVIDA"]))
chk = pd.concat([g.sample(min(AC["checagem_por_pasta"], len(g)), random_state=7) for _, g in c2.groupby("pasta")])
chk["pasta_kml"] = "CONFERIR"
(D / "amostras_usuario" / "rodada1_checagem.kml").write_text(
    kml(chk.sample(frac=1, random_state=3), "MANGA checagem rodada 1 — mover para MANGA_ok / NAO_MANGA / DUVIDA", ["CONFERIR"]),
    encoding="utf-8")
chk[["id", "regiao", "pasta", "p_manga"]].to_csv(D / "rotulos" / "candidatos" / "rodada1_checagem_gabarito.csv", index=False)

# ---------------- 5. relatório ----------------
tab = cand.pivot_table(index="regiao", columns="pasta", values="id", aggfunc="count", fill_value=0)
area_ok = cand[cand.pasta == "MANGA_ok"].groupby("regiao").area_ha.sum()
tab["area_ok_ha"] = area_ok.reindex(tab.index).fillna(0).round(0)
tab["pam_2025_ha"] = [pam.get(int(r[1:8]), 0) for r in tab.index]
rep = RAIZ / "reports" / "A4a_classificacao_auto.md"
rep.write_text("\n".join([
    "# A4a — Classificação automática da rodada 1 (rótulo AUTOMÁTICO, não usar para avaliação)", "",
    f"Treino XGBoost (GPU, 4 classes): {len(y):,} px — manga {(y == 1).sum():,}, outros {(y == 0).sum():,}, caju {n_caju:,}, coco {n_coco:,}.",
    f"CV espacial ({TR['folds']} folds; blocos de 3 km em Livramento): **AUC manga {auc:.3f}**, Brier calibrado {brier:.3f}.", "",
    "## Confiabilidade (p calibrada fora do fold × fração real de manga)", "", conf.to_markdown(), "",
    "## Regras", "",
    f"- Pixel confiável: p ≥ {AC['p_ok']} (Chapada) ou p ≥ {AC['p_ok_regiao_nova']} e p_manga − max(p_caju, p_coco) ≥ "
    f"{AC['margem_caju_coco']} (regiões novas); abertura 1 px; manchas ≥ {OKP['componente_min_px']} px.",
    f"- MANGA_ok (opção A, por pixel): classe MapBiomas compatível {AC['classes_compativeis']}, parte confiável ≥ {OKP['area_min_ha']} ha "
    f"e ≥ {100 * OKP['frac_min']:.0f}% do candidato; **o polígono gravado é só o recorte confiável**; área ≤ {AC['teto_pam']} × PAM 2025.",
    f"- NAO_MANGA: p ≤ {AC['p_nao']}, ou controle negativo com p < {AC['p_nao_controle']}. DUVIDA: o resto.", "",
    "## Resultado por município", "", tab.to_markdown(), "",
    f"Total: {cand.pasta.value_counts().to_dict()}", "",
    f"KMZ: `{kmz}` · checagem: `{D / 'amostras_usuario' / 'rodada1_checagem.kml'}` "
    f"({len(chk)} candidatos sorteados, gabarito em `dados/rotulos/candidatos/rodada1_checagem_gabarito.csv`)."]), encoding="utf-8")
print(f"Pastas: {cand.pasta.value_counts().to_dict()}")
print(f"Área MANGA_ok: {cand.loc[cand.pasta == 'MANGA_ok', 'area_ha'].sum():,.0f} ha | checagem: {len(chk)} candidatos")
print(f"Relatório: {rep}")
