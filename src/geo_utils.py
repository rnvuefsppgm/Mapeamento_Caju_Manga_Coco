"""Utilidades vetoriais: leitura, validação, reprojeção e agrupamento espacial.

Conceito metodológico central (ETAPA 18 / PONTO CRÍTICO):
    Um shapefile de entrada pode ter DOIS papéis distintos que NÃO podem ser
    confundidos:
        (a) AOI de download  -> apenas delimita onde baixar imagens;
        (b) ground truth/label -> polígonos de caju CONFIRMADO.
    A função ``classify_vector_role`` apenas alerta; a decisão é do usuário e é
    controlada pelo parâmetro ``--role`` dos scripts.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import geopandas as gpd
import numpy as np
from shapely.geometry import box
from shapely.validation import make_valid

logger = logging.getLogger(__name__)


def read_vector(path: str | Path, target_crs: str | None = None) -> gpd.GeoDataFrame:
    """Lê um vetor (shp/gpkg/geojson), valida geometrias e reprojeta se preciso.

    Args:
        path: caminho do arquivo vetorial.
        target_crs: CRS de destino (ex.: 'EPSG:32724'). Se ``None``, mantém o CRS.

    Returns:
        GeoDataFrame com geometrias válidas.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Vetor não encontrado: {path}")
    gdf = gpd.read_file(path)
    logger.info("Lido %s: %d feições, CRS=%s", path.name, len(gdf), gdf.crs)

    if gdf.crs is None:
        raise ValueError(
            f"{path.name} não possui CRS definido. Defina o CRS antes de prosseguir."
        )

    gdf = validate_geometries(gdf)

    if target_crs and str(gdf.crs) != str(target_crs):
        logger.info("Reprojetando %s -> %s", gdf.crs, target_crs)
        gdf = gdf.to_crs(target_crs)
    return gdf


def validate_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Corrige geometrias inválidas e remove nulas/vazias.

    A ORDEM importa: primeiro descartamos geometrias NULAS (algumas feições do
    shapefile podem vir sem geometria) e vazias, porque ``make_valid(None)``
    quebra ('NoneType' object has no attribute 'is_valid'). Só então corrigimos
    as inválidas restantes.
    """
    # 1) remove geometrias NULAS primeiro (senão make_valid recebe None e quebra)
    before = len(gdf)
    gdf = gdf[gdf.geometry.notna()].copy()
    n_null = before - len(gdf)

    # 2) remove geometrias vazias (agora sem None, is_empty é seguro)
    before2 = len(gdf)
    gdf = gdf[~gdf.geometry.is_empty].copy()
    n_empty = before2 - len(gdf)
    if n_null or n_empty:
        logger.warning("Removidas %d geometrias nulas e %d vazias.", n_null, n_empty)

    # 3) corrige as inválidas restantes (self-intersections etc.)
    invalid = ~gdf.geometry.is_valid
    n_invalid = int(invalid.sum())
    if n_invalid:
        logger.warning("Corrigindo %d geometrias inválidas com make_valid.", n_invalid)
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].apply(make_valid)

    # 4) make_valid pode gerar geometrias vazias -> limpa de novo
    before3 = len(gdf)
    gdf = gdf[~gdf.geometry.is_empty].copy()
    if before3 != len(gdf):
        logger.warning("Removidas %d geometrias vazias após make_valid.",
                       before3 - len(gdf))

    return gdf.reset_index(drop=True)


def classify_vector_role(gdf: gpd.GeoDataFrame) -> str:
    """Heurística leve para lembrar o usuário do papel do vetor.

    NÃO decide nada — apenas registra um aviso. Muitas feições pequenas sugerem
    ground truth; poucas feições grandes sugerem AOI de download.
    """
    areas_ha = gdf.to_crs(gdf.estimate_utm_crs()).area / 1e4
    median_ha = float(np.median(areas_ha))
    if median_ha > 500:
        logger.warning(
            "Área mediana das feições ~%.0f ha: parece AOI de download, não label. "
            "Confirme com --role antes de rasterizar como caju.", median_ha,
        )
        return "likely_aoi"
    return "likely_ground_truth"


def total_bounds_bbox(gdf: gpd.GeoDataFrame, buffer_m: float = 0.0) -> List[float]:
    """Bounding box total [minx, miny, maxx, maxy] em WGS84 (para busca STAC).

    A busca STAC opera em EPSG:4326. Faz-se um buffer opcional (em metros no CRS
    métrico do gdf) antes de reprojetar para lon/lat.
    """
    g = gdf
    if buffer_m > 0:
        g = gdf.copy()
        g["geometry"] = g.geometry.buffer(buffer_m)
    g_wgs = g.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = g_wgs.total_bounds
    return [float(minx), float(miny), float(maxx), float(maxy)]


def group_by_spatial_blocks(
    gdf: gpd.GeoDataFrame, block_size_km: float
) -> gpd.GeoDataFrame:
    """Atribui a cada feição um ``block_id`` com base em uma grade regular.

    Fundamental para o *spatial split*: feições no mesmo bloco nunca são divididas
    entre treino/validação/teste, prevenindo *spatial leakage* (ETAPA 21).

    Requer que ``gdf`` esteja em CRS métrico.
    """
    if gdf.crs is None or gdf.crs.is_geographic:
        raise ValueError("group_by_spatial_blocks requer CRS métrico (UTM).")
    size_m = block_size_km * 1000.0
    cent = gdf.geometry.centroid
    ix = np.floor(cent.x / size_m).astype(int)
    iy = np.floor(cent.y / size_m).astype(int)
    gdf = gdf.copy()
    gdf["block_id"] = [f"blk_{a}_{b}" for a, b in zip(ix, iy)]
    logger.info("Atribuídos %d blocos espaciais (%.0f km).",
                gdf["block_id"].nunique(), block_size_km)
    return gdf


def cluster_bboxes_for_query(
    gdf: gpd.GeoDataFrame, cluster_size_km: float = 40.0
) -> List[Tuple[str, List[float], gpd.GeoDataFrame]]:
    """Agrupa feições próximas em bboxes de consulta para evitar buscas redundantes.

    Retorna lista de (cluster_id, bbox_wgs84, subset_gdf). Cada bbox cobre um
    conjunto de feições vizinhas, reduzindo o número de chamadas ao STAC
    (ETAPA 3: "quando houver muitos polígonos, evite consultas redundantes").
    """
    grouped = group_by_spatial_blocks(gdf, cluster_size_km)
    clusters: List[Tuple[str, List[float], gpd.GeoDataFrame]] = []
    for block_id, sub in grouped.groupby("block_id"):
        bbox = total_bounds_bbox(sub)
        clusters.append((str(block_id), bbox, sub.copy()))
    logger.info("Gerados %d clusters de consulta STAC.", len(clusters))
    return clusters


def geometry_grid_from_bounds(
    bounds: Tuple[float, float, float, float], resolution: float
) -> Dict[str, Any]:
    """Deriva parâmetros de grade (largura/altura/transform) para um bbox métrico.

    Usado para definir o grid de referência a 10 m no qual S2 e S1 são alinhados.
    """
    from rasterio.transform import from_origin

    minx, miny, maxx, maxy = bounds
    # Alinha os limites a múltiplos da resolução (grid "snap").
    minx = np.floor(minx / resolution) * resolution
    miny = np.floor(miny / resolution) * resolution
    maxx = np.ceil(maxx / resolution) * resolution
    maxy = np.ceil(maxy / resolution) * resolution
    width = int(round((maxx - minx) / resolution))
    height = int(round((maxy - miny) / resolution))
    transform = from_origin(minx, maxy, resolution, resolution)
    return {"width": width, "height": height, "transform": transform,
            "bounds": (minx, miny, maxx, maxy)}
