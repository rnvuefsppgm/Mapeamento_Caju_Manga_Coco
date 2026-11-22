"""03 — Tabela final por município (área mapeada de caju/manga/coco × PAM 2023–25 × Censo 2017) e documento das classes.
Entradas: saidas/tabelas/areas_municipio_classe.csv e cobertura_municipios.csv (02), dados/pam_censo/pam_censo_3culturas.csv (01).
Saídas: saidas/tabelas/tabela_final_municipios.csv e docs/CLASSES_MAPA_UNICO.md
"""
import pandas as pd

from common import cfg, p

c = cfg()
CULT = c["culturas"]
a = pd.read_csv(p("saidas", "tabelas", "areas_municipio_classe.csv"), dtype={"classe": str})
cob = pd.read_csv(p("saidas", "tabelas", "cobertura_municipios.csv"))
pc = pd.read_csv(p("dados", "pam_censo", "pam_censo_3culturas.csv"))
mun = a.groupby("cod").municipio.first()
uf = pc.drop_duplicates("cod").set_index("cod").nome.str.extract(r"\((\w\w)\)")[0]
T = pd.DataFrame({"cod": mun.index, "municipio": mun.values}).merge(cob, on="cod")
T.insert(2, "uf", T.cod.map(uf))
T["area_municipio_ha"] = T.cod.map(a[~a.classe.str.startswith("disputa") & (a.classe != "255")].groupby("cod").area_ha.sum())
for k, v in CULT.items():
    m = a[a.classe == str(v["codigo"])].groupby("cod").area_ha.sum()
    s = pc[pc.cultura == k].set_index("cod")
    T[f"{k}_mapa_ha"] = T.cod.map(m).fillna(0).where(T[f"mapeado_{k}"])     # vazio = cultura não mapeada no município
    T[f"{k}_pam_ha"] = T.cod.map(s.pam_area_destinada_ha)
    T[f"{k}_censo_ha"] = T.cod.map(s.censo_area_existente_ha)
    T[f"{k}_mapa_pam"] = T[f"{k}_mapa_ha"] / T[f"{k}_pam_ha"]
T["disputa_ha"] = T.cod.map(a[a.classe.str.startswith("disputa")].groupby("cod").area_ha.sum()).fillna(0)
T = T.sort_values(["uf", "municipio"])
T.round(2).to_csv(p("saidas", "tabelas", "tabela_final_municipios.csv"), index=False)


def n(x, d=0):
    return "—" if pd.isna(x) else f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


lin = ["# Mapa único 2025 — caju, manga e coco sobre MapBiomas Coleção 11", "",
       "Arquivo: `saidas/mapa/culturas_2025_mapbiomas_c11.tif` — GeoTIFF uint8, EPSG:32724 (UTM 24S), 10 m, paleta embutida.", "",
       "## Valores das classes", "", "| Valor | Classe | Cor | Origem |", "|---|---|---|---|"]
for k, v in CULT.items():
    lin.append(f"| **{v['codigo']}** | {k.capitalize()} | `{v['cor']}` | mapa final do projeto ({v['classe'].split('/')[1]}), "
               f"limiar {v['limiar']}" + (" (0,30 no sertão CE/RN, fusão U-Net+XGB)" if k == "caju" else "") + " |")
lin += ["| 1–91 | Classes da legenda MapBiomas Coleção 11 (ex.: 3 Formação Florestal, 4 Savânica, 15 Pastagem, 21 Mosaico de "
        "Usos, 48 Outras Lavouras Perenes) | cores oficiais | MapBiomas 2025, 30 m reamostrado a 10 m (vizinho mais próximo) |",
        "| 0 | Sem dado no MapBiomas | — | — |",
        "| 255 | Fora dos municípios mapeados (NoData) | transparente | — |", "",
        "Os valores 97, 98 e 99 **não constam** da legenda da Coleção 11 (conferido nas 5 abas da planilha "
        "`Legenda Coleção 11 - Descrição Detalhada.xlsx`); o mapa pode ser lido junto com qualquer produto MapBiomas.", "",
        "## Regras", "",
        "1. **Fundo**: classe MapBiomas 2025. Onde um modelo marcou caju, manga ou coco, a cultura substitui o fundo.",
        "2. **Sobreposição** (pixel marcado por 2+ culturas): vence a maior margem normalizada `(p − limiar)/(1 − limiar)`, "
        "com `p` = probabilidade do modelo de cada cultura. A probabilidade bruta não é usada porque os limiares diferem "
        "(caju 0,30/0,40, manga 0,55, coco 0,65). As disputas ficam em `saidas/auxiliar/disputas_2025.tif` "
        "(soma de bits: 1 caju, 2 manga, 4 coco; 3 = caju×manga, 5 = caju×coco, 6 = manga×coco).",
        "3. **Cobertura**: cada cultura só foi mapeada nos seus municípios (caju 25, manga 25, coco 15). Num município onde a "
        "cultura não foi mapeada, a tabela mostra \"—\" (não é ausência). Ver `saidas/tabelas/cobertura_municipios.csv`.", "",
        "## Área por município (ha)", "",
        "Mapa = área classificada; PAM = área destinada à colheita, média 2023–25 (IBGE 1613); Censo = área existente 2017 "
        "(IBGE 6955). Caju: PAM e Censo de castanha de caju. \"—\" = cultura não mapeada no município ou sem dado.", "",
        "| UF | Município | Caju mapa | Caju PAM | Caju Censo | Manga mapa | Manga PAM | Manga Censo | Coco mapa | Coco PAM | Coco Censo | Disputa |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in T.itertuples():
    lin.append(f"| {r.uf} | {r.municipio} | " + " | ".join(n(getattr(r, f"{k}_{x}")) for k in CULT for x in ("mapa_ha", "pam_ha", "censo_ha"))
               + f" | {n(r.disputa_ha)} |")
tot = {k: (T[f"{k}_mapa_ha"].sum(), T.loc[T[f"mapeado_{k}"], f"{k}_pam_ha"].sum(), T.loc[T[f"mapeado_{k}"], f"{k}_censo_ha"].sum()) for k in CULT}
lin.append("| | **Total (municípios mapeados da cultura)** | " + " | ".join(f"**{n(x)}**" for k in CULT for x in tot[k]) + f" | **{n(T.disputa_ha.sum())}** |")
lin += ["", "Tabela completa (com razões mapa/PAM e área do município): `saidas/tabelas/tabela_final_municipios.csv`."]
p("docs", "CLASSES_MAPA_UNICO.md").write_text("\n".join(lin), encoding="utf-8")
for k in CULT:
    print(f"{k}: mapa {n(tot[k][0])} ha | PAM {n(tot[k][1])} | Censo {n(tot[k][2])}")
print(f"disputas {n(T.disputa_ha.sum())} ha | {len(T)} municípios -> docs/CLASSES_MAPA_UNICO.md")
