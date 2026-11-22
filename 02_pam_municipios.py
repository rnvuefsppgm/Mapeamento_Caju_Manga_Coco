"""02 — PAM manga (SIDRA 1613, Manga = 2737) 2023-2025 por município + seleção dos municípios a mapear.

Regra (igual ao caju): municípios do bioma configurado com manga na PAM, ordenados pela produção média
2023-2025; ficam os N maiores + os de `incluir_sempre` (amostras de campo). Limites de BR_Municipios_2024.
Saídas: dados/pam/pam_manga_2023_2025.{json,csv}, dados/pam/ranking_manga.csv,
        dados/pam/municipios_selecionados.{csv,gpkg}, reports/02_pam_municipios.md
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.config import load_config, resolve_path, setup_logging

cfg = load_config(Path(__file__).parent / "config/config.yaml")
log = setup_logging("02_pam_municipios", cfg)
M, P, S = cfg["manga"], cfg["manga"]["pam"], cfg["manga"]["selecao"]
NOMES = {"216": "area_colhida_ha", "214": "prod_t", "2313": "area_dest_ha"}


def baixar(out: Path):
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    url = (f"https://servicodados.ibge.gov.br/api/v3/agregados/{P['tabela']}/periodos/"
           f"{'|'.join(map(str, P['anos']))}/variaveis/{'|'.join(map(str, P['variaveis']))}"
           f"?localidades=N6[all]&classificacao={P['classificacao']}[{P['categoria_manga']}]")
    with urllib.request.urlopen(url, timeout=300) as r:
        js = json.loads(r.read().decode("utf-8"))
    out.write_text(json.dumps(js, ensure_ascii=False), encoding="utf-8")
    return js


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0  # "-", "...", "X" -> 0


def main():
    d = resolve_path(cfg, "pam")
    js = baixar(d / "pam_manga_2023_2025.json")
    rows = []
    for var in js:
        for s in var["resultados"][0]["series"]:
            for ano, v in s["serie"].items():
                rows.append({"cod": int(s["localidade"]["id"]), "nome": s["localidade"]["nome"],
                             "var": NOMES[str(var["id"])], "ano": int(ano), "v": num(v)})
    df = pd.DataFrame(rows).pivot_table(index=["cod", "nome"], columns=["var", "ano"], values="v", aggfunc="sum")
    df.columns = [f"{a}_{b}" for a, b in df.columns]
    df = df.reset_index()
    anos = P["anos"]
    df["prod_med_t"] = df[[f"prod_t_{a}" for a in anos]].mean(axis=1)
    df["area_med_ha"] = df[[f"area_colhida_ha_{a}" for a in anos]].mean(axis=1)
    df["uf"] = df.nome.str[-2:]
    bio = pd.read_csv(M["fontes"]["bioma_csv"], sep=";", encoding="utf-8-sig")
    bio.columns = ["cod", "nm", "uf_b", "bioma"]
    bio["cod"] = pd.to_numeric(bio.cod, errors="coerce")
    bio = bio.dropna(subset=["cod"]).astype({"cod": int})
    df = df.merge(bio[["cod", "bioma"]], on="cod", how="left")
    df = df[df.prod_med_t > 0].sort_values("prod_med_t", ascending=False)
    df.to_csv(d / "pam_manga_2023_2025.csv", index=False, encoding="utf-8-sig")

    rk = df if not S["bioma"] else df[df.bioma == S["bioma"]].copy()
    tot_b, tot_br = rk.prod_med_t.sum(), df.prod_med_t.sum()
    rk["part_%"] = 100 * rk.prod_med_t / tot_b
    rk["acum_%"] = rk["part_%"].cumsum()
    rk["rank"] = range(1, len(rk) + 1)
    rk.to_csv(d / "ranking_manga.csv", index=False, encoding="utf-8-sig")
    sel = pd.concat([rk.head(S["n_municipios"]), df[df.cod.isin(S["incluir_sempre"])]]).drop_duplicates("cod")
    sel["motivo"] = ["ranking" if c in set(rk.head(S["n_municipios"]).cod) else "amostras" for c in sel.cod]
    mun = gpd.read_file(M["fontes"]["municipios_shp"])
    mun["cod"] = mun.CD_MUN.astype(int)
    g = mun[["cod", "NM_MUN", "SIGLA_UF", "AREA_KM2", "geometry"]].merge(sel, on="cod")
    g.to_file(d / "municipios_selecionados.gpkg", driver="GPKG")
    sel.to_csv(d / "municipios_selecionados.csv", index=False, encoding="utf-8-sig")

    a = anos[-1]
    linhas = "".join(
        f"| {r.get('rank', '-')} | {r.nome} | {r.bioma} | {r.prod_med_t:,.0f} | {r[f'area_colhida_ha_{a}']:,.0f} | "
        f"{r.get('acum_%', float('nan')):.1f} | {r.motivo} |\n" for _, r in sel.iterrows())
    (resolve_path(cfg, "reports") / "02_pam_municipios.md").write_text(
        f"# 02 — PAM manga {anos[0]}–{anos[-1]} e municípios selecionados\n\n"
        f"Municípios com manga na PAM (Brasil): {len(df)}; no bioma {S['bioma']}: {len(rk)} "
        f"({100 * tot_b / tot_br:.1f}% da produção nacional média).\n"
        f"Selecionados: {len(sel)} ({S['n_municipios']} maiores + amostras) = "
        f"{100 * sel.prod_med_t.sum() / tot_b:.1f}% da produção do bioma; "
        f"área colhida {a}: {sel[f'area_colhida_ha_{a}'].sum():,.0f} ha.\n\n"
        f"| # | Município | Bioma | Prod. média (t) | Área colhida {a} (ha) | Acum. % | Motivo |\n"
        "|---|---|---|---|---|---|---|\n" + linhas, encoding="utf-8")
    log.info(f"RESUMO: {len(df)} municípios com manga; {len(rk)} na {S['bioma']}; selecionados {len(sel)} "
             f"({100 * sel.prod_med_t.sum() / tot_b:.1f}% da produção do bioma; {len(g)} com limite)")


if __name__ == "__main__":
    main()
