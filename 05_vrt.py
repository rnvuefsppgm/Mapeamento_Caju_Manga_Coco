"""05 — Mosaico virtual (VRT) por região a partir dos tiles COG do DatensEE (porte do caju v8/03_mosaic_vrt.py).
dados/embeddings/<regiao>.vrt aponta para os tiles (mesma grade EPSG:32724 10 m), sem duplicar dados.
"""
from __future__ import annotations

import glob
from pathlib import Path

import rasterio

from src.config import load_config, resolve_path

DT = {"uint8": "Byte", "int16": "Int16", "float32": "Float32"}


def build_vrt(tile_dir: Path) -> Path | None:
    tiles = sorted(glob.glob(str(tile_dir / "*.tif")))
    if not tiles:
        return None
    infos = []
    for t in tiles:
        with rasterio.open(t) as d:
            infos.append((t, d.bounds, d.width, d.height, d.count, d.dtypes[0], d.nodata, d.crs, d.res))
    res = infos[0][8][0]
    x0 = min(i[1].left for i in infos); y1 = max(i[1].top for i in infos)
    x1 = max(i[1].right for i in infos); y0 = min(i[1].bottom for i in infos)
    W, H = round((x1 - x0) / res), round((y1 - y0) / res)
    nb, dt, nd, crs = infos[0][4], infos[0][5], infos[0][6], infos[0][7]
    out = tile_dir.with_suffix(".vrt")
    L = [f'<VRTDataset rasterXSize="{W}" rasterYSize="{H}">', f"<SRS>{crs.to_wkt()}</SRS>",
         f"<GeoTransform>{x0}, {res}, 0, {y1}, 0, {-res}</GeoTransform>"]
    for b in range(1, nb + 1):
        L.append(f'<VRTRasterBand dataType="{DT[dt]}" band="{b}">')
        if nd is not None:
            L.append(f"<NoDataValue>{nd}</NoDataValue>")
        for t, bd, w, h, *_ in infos:
            xo, yo = round((bd.left - x0) / res), round((y1 - bd.top) / res)
            rel = Path(t).relative_to(tile_dir.parent).as_posix()
            L.append(f'<SimpleSource><SourceFilename relativeToVRT="1">{rel}</SourceFilename>'
                     f"<SourceBand>{b}</SourceBand>"
                     f'<SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>'
                     f'<DstRect xOff="{xo}" yOff="{yo}" xSize="{w}" ySize="{h}"/></SimpleSource>')
        L.append("</VRTRasterBand>")
    L.append("</VRTDataset>")
    out.write_text("\n".join(L), encoding="utf-8")
    return out


if __name__ == "__main__":
    cfg = load_config(Path(__file__).parent / "config/config.yaml")
    base = resolve_path(cfg, "embeddings")
    ok = [d.name for d in sorted(p for p in base.iterdir() if p.is_dir()) if build_vrt(d)]
    print(f"VRT: {len(ok)} regiões")
