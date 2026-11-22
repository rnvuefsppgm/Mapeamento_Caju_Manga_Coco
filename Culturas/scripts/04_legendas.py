"""04 — Legendas do mapa único para ArcMap/ArcGIS: cores e nomes a partir da legenda MapBiomas Coleção 11 (planilha)
+ classes novas (97 caju, 98 manga, 99 coco). Saídas em legendas/:
  culturas_2025_mapbiomas_c11.clr            colormap ArcMap (valor R G B)
  culturas_2025_mapbiomas_c11_nomes.clr      colormap com nome (valor R G B nome) — ArcGIS Pro / QGIS
  legenda_classes.csv                        código, nome (pt/en), cor hex/RGB, área no mapa (ha)
  culturas_2025_mapbiomas_c11.tif.vat.dbf    tabela de atributos (Value, Count, Classe, R, G, B) — nomes no ArcMap;
                                             cópia também ao lado do raster em saidas/mapa/
"""
import re
import shutil
import struct
import zipfile

import pandas as pd

from common import cfg, p

c = cfg()
NOVAS = {"caju": ("Caju", "Cashew"), "manga": ("Manga", "Mango"), "coco": ("Coco-da-baía", "Coconut")}


def legenda_mb(xlsx):
    z = zipfile.ZipFile(xlsx)
    ss = [re.sub("<[^>]+>", "", s) for s in re.findall(r"<si>(.*?)</si>", z.read("xl/sharedStrings.xml").decode("utf8"), re.S)]
    L = {}
    for row in re.findall(r"<row[^>]*>(.*?)</row>", z.read("xl/worksheets/sheet1.xml").decode("utf8"), re.S):
        v = []
        for attr, inner in re.findall(r"<c r=\"[A-Z]+\d+\"([^>]*?)(?:/>|>(.*?)</c>)", row, re.S):
            m = re.search(r"<v>(.*?)</v>", inner or "")
            if m:
                v.append((ss[int(m.group(1))] if 't="s"' in attr else m.group(1)).strip())
        i = next((k for k, x in enumerate(v) if re.fullmatch(r"\d+(\.0)?", x) and k >= 2), None)
        if i is None or i + 1 >= len(v) or not re.fullmatch(r"#[0-9a-fA-F]{6}", v[i + 1]):
            continue
        cod = int(float(v[i]))
        tira = lambda s: re.sub(r"^\s*[\d.]+\s*", "", s).strip()   # remove numeração "3.2.2.1."
        L[cod] = (tira(v[i - 2]), tira(v[i - 1]), v[i + 1].lower())
    return L


L = legenda_mb(c["legenda_mapbiomas"])
for k, v in c["culturas"].items():
    L[v["codigo"]] = (*NOVAS[k], v["cor"].lower())
L.setdefault(0, ("Sem dado / não observado", "No data", "#ffffff"))
a = pd.read_csv(p("saidas", "tabelas", "areas_municipio_classe.csv"), dtype={"classe": str})
area = a[a.classe.str.fullmatch(r"\d+")].assign(v=lambda d: d.classe.astype(int)).groupby("v").area_ha.sum()
rows = []
for cod in sorted(L):
    pt, en, hx = L[cod]
    r, g, b = (int(hx[i:i + 2], 16) for i in (1, 3, 5))
    rows.append(dict(codigo=cod, nome=pt, nome_en=en, cor_hex=hx, R=r, G=g, B=b,
                     area_ha=round(float(area.get(cod, 0)), 2), nova=cod in (97, 98, 99)))
d = pd.DataFrame(rows)
d = d[(d.area_ha > 0) | d.nova]          # só classes presentes no mapa (+ as 3 novas)
out = p("legendas", "x").parent
d.to_csv(out / "legenda_classes.csv", index=False, encoding="utf-8-sig")
base = "culturas_2025_mapbiomas_c11"
(out / f"{base}.clr").write_text("\n".join(f"{r.codigo} {r.R} {r.G} {r.B}" for r in d.itertuples()) + "\n", encoding="cp1252")
(out / f"{base}_nomes.clr").write_text("\n".join(f"{r.codigo} {r.R} {r.G} {r.B} {r.nome}" for r in d.itertuples()) + "\n", encoding="cp1252")


def dbf(path, recs, campos):
    """dBASE III simples. campos: [(nome, 'N'|'C', tamanho, decimais)]"""
    with open(path, "wb") as f:
        f.write(struct.pack("<BBBBIHH17xB2x", 3, 126, 11, 22, len(recs), 32 + 32 * len(campos) + 1, 1 + sum(t for _, _, t, _ in campos), 0x57))
        for n, tp, t, dc in campos:
            f.write(struct.pack("<11sc4xBB14x", n.encode(), tp.encode(), t, dc))
        f.write(b"\r")
        for r in recs:
            f.write(b" ")
            for (n, tp, t, dc), v in zip(campos, r):
                s = (f"{v:>{t}.{dc}f}" if dc else f"{int(v):>{t}d}") if tp == "N" else f"{str(v):<{t}}"[:t]
                f.write(s.encode("cp1252", "replace"))
        f.write(b"\x1a")


cnt = (d.area_ha * 100).round().astype("int64")
recs = [(r.codigo, n, r.nome[:60], r.R, r.G, r.B) for r, n in zip(d.itertuples(), cnt)]
campos = [("Value", "N", 4, 0), ("Count", "N", 12, 0), ("Classe", "C", 60, 0), ("Red", "N", 3, 0), ("Green", "N", 3, 0), ("Blue", "N", 3, 0)]
vat = out / f"{base}.tif.vat.dbf"
dbf(vat, recs, campos)
shutil.copy(vat, p("saidas", "mapa", f"{base}.tif.vat.dbf"))
print(f"{len(d)} classes ({int(d.nova.sum())} novas) -> {out}")
print(d[["codigo", "nome", "cor_hex", "area_ha"]].to_string(index=False))

# --- README da pasta legendas: tabela de todas as classes com amostra de cor (PNG local em cores/)
from PIL import Image

(out / "cores").mkdir(exist_ok=True)
for r in d.itertuples():
    Image.new("RGB", (16, 16), (r.R, r.G, r.B)).save(out / "cores" / f"{r.codigo}.png")
rd = out / "README.md"
txt = rd.read_text(encoding="utf-8").split("\n## Todas as classes")[0].rstrip()
lin = ["", "", "## Todas as classes", "", "| Cor | Código | Classe | Class (en) | Hex | R, G, B | Área no mapa (ha) |", "|---|---|---|---|---|---|---|"]
lin += [f"| ![{r.cor_hex}](cores/{r.codigo}.png) | **{r.codigo}** | {'**' + r.nome + '**' if r.nova else r.nome} | {r.nome_en} | "
        f"`{r.cor_hex}` | {r.R}, {r.G}, {r.B} | {f'{r.area_ha:,.0f}'.replace(',', '.')} |"
        for r in d.itertuples()]
rd.write_text(txt + "\n".join(lin) + "\n\nEm negrito: classes novas (não constam da legenda MapBiomas Coleção 11).\n", encoding="utf-8")
print(f"README com {len(d)} classes -> {rd}")
