"""Pipeline compartilhado: AOI -> composto multissensor fundido.

Esta é a função ÚNICA que transforma uma AOI (geometria + janela temporal) no
stack de features fundido, no grid de 10 m. Tanto o script de treino (01) quanto
o de predição (04) a chamam, garantindo que treinamento e inferência usem
EXATAMENTE o mesmo pré-processamento (ETAPA 34: "não duplique lógica").
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import xarray as xr
from pyproj import CRS

from . import sar_features as sarf
from . import sensor_fusion as fusion
from . import spectral_indices as sindex
from . import stac_utils as stac
from . import temporal_composites as tcomp
from . import sentinel1_preprocessing as s1p
from . import sentinel2_preprocessing as s2p
from .geo_utils import geometry_grid_from_bounds, total_bounds_bbox

logger = logging.getLogger(__name__)


def _epsg_from_crs(crs_str: str) -> int:
    return CRS.from_user_input(crs_str).to_epsg()


def build_fused_composite(
    catalog,
    aoi: gpd.GeoDataFrame,
    cfg: Dict[str, Any],
) -> Tuple[Optional[xr.DataArray], Dict[str, Any]]:
    """Constrói o composto fundido para uma AOI.

    Args:
        catalog: cliente STAC (assinado).
        aoi: GeoDataFrame (1+ feições) já no CRS de trabalho (métrico).
        cfg: configuração.

    Returns:
        (stack_fundido[band,y,x] ou None se sem dados, dict de proveniência/grid).
    """
    epsg = _epsg_from_crs(cfg["project"]["crs_working"])
    resolution = cfg["harmonization"]["resolution"]
    modality = cfg["experiment"]["modality"]

    # Grid de referência (10 m) a partir dos bounds métricos da AOI.
    bounds_m = tuple(aoi.total_bounds)
    grid = geometry_grid_from_bounds(bounds_m, resolution)
    bounds = grid["bounds"]
    bbox_wgs = total_bounds_bbox(aoi, buffer_m=resolution * 2)

    dr = f"{cfg['temporal']['start_date']}/{cfg['temporal']['end_date']}"
    prov: Dict[str, Any] = {"grid": grid, "s2_items": [], "s1_items": [],
                            "orbit_direction": ""}

    s2_bands_comp = s2_idx_comp = s1_comp = sar_comp = None

    # ---------------- Sentinel-2 ---------------- #
    if modality in ("s2", "fusion"):
        s2_items = stac.search_items(
            catalog, cfg["sentinel2"]["collection"], bbox_wgs, dr,
            query={"eo:cloud_cover": {"lt": cfg["sentinel2"]["max_scene_cloud_cover"]}},
        )
        s2_items = stac.deduplicate_items(s2_items)
        s2_items = stac.deduplicate_s2_acquisitions(s2_items)  # v6
        prov["s2_items"] = [it.id for it in s2_items]
        if s2_items:
            stack = s2p.load_s2_stack(
                s2_items, cfg["sentinel2"]["bands"], bounds, epsg, resolution,
                cfg["sentinel2"]["scl_asset"])
            masked = s2p.apply_scl_mask_and_scale(stack, cfg["sentinel2"]["bands"], cfg)
            masked = s2p.select_valid_scenes(
                masked, cfg["sentinel2"]["min_valid_fraction_aoi"])
            if masked.sizes.get("time", 0) > 0:
                s2_bands_comp = tcomp.temporal_composite(
                    masked, cfg["temporal"]["s2_reducers"],
                    cfg["temporal"]["min_observations"])
                if cfg["experiment"]["use_indices"]:
                    # índices calculados sobre o composto de MEDIANA das bandas base
                    median_ref = _median_bands(masked, cfg["sentinel2"]["bands"])
                    s2_idx_comp = sindex.compute_indices(
                        median_ref, cfg["spectral_indices"])

    # ---------------- Sentinel-1 ---------------- #
    if modality in ("s1", "fusion"):
        query = None
        if cfg["sentinel1"]["orbit_direction"] not in ("any", None):
            query = {"sat:orbit_state": {"eq": cfg["sentinel1"]["orbit_direction"]}}
        s1_items = stac.search_items(
            catalog, cfg["sentinel1"]["collection"], bbox_wgs, dr, query=query)
        s1_items = stac.deduplicate_items(s1_items)

        if cfg["sentinel1"].get("prefer_single_orbit") and \
                cfg["sentinel1"]["orbit_direction"] in ("any", None):
            dom = stac.dominant_orbit_direction(s1_items)
            if dom:
                s1_items = [it for it in s1_items
                            if (it.properties.get("sat:orbit_state") == dom)]
                prov["orbit_direction"] = dom
        prov["s1_items"] = [it.id for it in s1_items]

        if s1_items:
            stack = s1p.load_s1_stack(
                s1_items, cfg["sentinel1"]["polarizations"], bounds, epsg, resolution)
            s1_db = s1p.preprocess_s1(stack, to_db=cfg["sentinel1"]["to_db"])
            s1_comp_full = tcomp.temporal_composite(
                s1_db, cfg["temporal"]["s1_reducers"],
                cfg["temporal"]["min_observations"])
            s1_comp = s1_comp_full
            if cfg["experiment"]["use_sar_features"]:
                med = _median_bands(s1_db, cfg["sentinel1"]["polarizations"])
                sar_comp = sarf.compute_sar_features(med, cfg)

    # ---------------- Fusão ---------------- #
    have_s2 = s2_bands_comp is not None
    have_s1 = s1_comp is not None
    if modality == "s2" and not have_s2:
        return None, prov
    if modality == "s1" and not have_s1:
        return None, prov
    if modality == "fusion" and not (have_s2 and have_s1):
        logger.warning("Fusão requer ambos sensores; faltou %s.",
                       "S2" if not have_s2 else "S1")
        return None, prov

    stack = fusion.build_feature_stack(
        modality, s2_bands=s2_bands_comp, s2_indices=s2_idx_comp,
        s1_bands=s1_comp, sar_features=sar_comp)
    prov["band_names"] = stack.attrs.get("band_names", [])
    return stack, prov


def _median_bands(masked: xr.DataArray, bands: List[str]) -> xr.DataArray:
    """Composto de mediana temporal das bandas base (para calcular índices)."""
    med = masked.median(dim="time", skipna=True)
    return med.assign_coords(band=list(bands))
