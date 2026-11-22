"""00 — Status compacto do projeto manga (uma linha por etapa). Rodar sempre antes de qualquer atividade."""
from pathlib import Path

R = Path(__file__).parent


def n(pat):
    return len(list(R.glob(pat)))


def gb(d):
    p = R / d
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9 if p.exists() else 0


def last(log):
    p = R / "logs" / log
    if not p.exists():
        return "-"
    ls = [x for x in p.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
    s = ls[-1] if ls else "-"
    return s.split("RESUMO", 1)[-1].lstrip(": ")[:100]  # sem data/nível do logging


regs = (R / "dados/aoi/regioes.txt")
nreg = len([x for x in regs.read_text().split() if x]) if regs.exists() else 0
linhas = [
    ("01 amostras", (R / "dados/amostras/manga_poligonos.gpkg").exists(), last("01_amostras.log")),
    ("02 PAM/municípios", (R / "dados/pam/municipios_selecionados.csv").exists(), last("02_pam_municipios.log")),
    ("03 regiões", nreg > 0, f"{nreg} regiões"),
    ("04 embeddings", n("dados/embeddings/*.vrt") >= nreg > 0,
     f"{n('dados/embeddings/*/*.tif')} tiles, {gb('dados/embeddings'):.1f} GB, {n('dados/embeddings/*.vrt')} VRT | {last('04_export.log')}"),
    ("06 MapBiomas", n("dados/mapbiomas/*_mb.tif") >= nreg > 0, f"{n('dados/mapbiomas/*_mb.tif')} recortes | {last('06_mapbiomas.log')}"),
    ("07 rótulos campo", n("dados/rotulos/*_campo.tif") > 0, last("07_rotulos_campo.log")),
    ("08 chips", n("dados/chips/*.npz") > 0, f"{n('dados/chips/*.npz')} chips"),
    ("09 cache", (R / "dados/cache/features_u8.npy").exists(), f"{gb('dados/cache'):.1f} GB"),
    ("10+ treino", n("modelos/**/*.pt") > 0, f"{n('modelos/**/*.pt')} pesos"),
]
for nome, ok, info in linhas:
    print(f"{'OK ' if ok else '.. '} {nome:<18} {info}")
