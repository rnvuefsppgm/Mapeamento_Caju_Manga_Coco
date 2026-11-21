"""Consultas STAC eficientes, deduplicação de Items e registro de metadados.

Estratégia (ETAPA 3):
    * consulta por bbox de clusters de feições vizinhas (não uma consulta por
      polígono, evitando redundância);
    * deduplicação de STAC Items entre clusters;
    * registro completo de proveniência por Item (ETAPA 45).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd
from pystac_client import Client

from .planetary_computer_utils import with_retry

logger = logging.getLogger(__name__)


@with_retry(max_retries=5, backoff_base=2.0)
def search_items(
    catalog: Client,
    collection: str,
    bbox: List[float],
    datetime_range: str,
    query: Dict[str, Any] | None = None,
    max_items: int | None = None,
) -> List[Any]:
    """Executa uma busca STAC e devolve a lista de Items (com retry).

    Args:
        catalog: cliente STAC (já com assinatura SAS via modifier).
        collection: ID da coleção (ex.: 'sentinel-2-l2a').
        bbox: [minx, miny, maxx, maxy] em EPSG:4326.
        datetime_range: 'YYYY-MM-DD/YYYY-MM-DD'.
        query: filtros por propriedade (ex.: {'eo:cloud_cover': {'lt': 60}}).
        max_items: limite opcional de Items.

    Returns:
        Lista de ``pystac.Item``.
    """
    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=datetime_range,
        query=query,
        max_items=max_items,
    )
    items = list(search.items())
    logger.info("STAC %s | bbox=%s | %s -> %d items",
                collection, [round(b, 3) for b in bbox], datetime_range, len(items))
    return items


def deduplicate_items(items: List[Any]) -> List[Any]:
    """Remove Items duplicados por ID, mantendo o primeiro de cada ID."""
    seen: set[str] = set()
    unique: List[Any] = []
    for it in items:
        if it.id not in seen:
            seen.add(it.id)
            unique.append(it)
    if len(unique) != len(items):
        logger.info("Deduplicação STAC: %d -> %d items.", len(items), len(unique))
    return unique


def deduplicate_s2_acquisitions(items: List[Any]) -> List[Any]:
    """v6: remove cenas S2 repetidas (mesma aquisição, processamentos diferentes).

    O Planetary Computer publica a cena original e a reprocessada (ex.:
    ``..._T24MYV_20231027T163319`` e ``..._T24MYV_20240830T115126``). Sem esta
    limpeza a mesma data entra duas vezes na mediana/percentis. Chave = ID sem o
    último segmento (data de processamento); mantém o processamento mais recente.
    """
    best: Dict[str, Any] = {}
    for it in items:
        parts = it.id.split("_")
        key, proc = "_".join(parts[:-1]), parts[-1]
        if key not in best or proc > best[key].id.split("_")[-1]:
            best[key] = it
    unique = list(best.values())
    if len(unique) != len(items):
        logger.info("Dedup S2 por aquisição: %d -> %d cenas.", len(items), len(unique))
    return unique


def item_provenance(item: Any, collection: str, sensor: str,
                    assets_used: List[str]) -> Dict[str, Any]:
    """Extrai metadados de proveniência de um Item (ETAPA 45).

    Campos ausentes viram ``None`` — coleções diferentes expõem propriedades
    diferentes, então nunca se assume presença.
    """
    props = item.properties
    return {
        "stac_item_id": item.id,
        "collection": collection,
        "sensor": sensor,
        "acquisition_date": props.get("datetime"),
        "eo_cloud_cover": props.get("eo:cloud_cover"),
        "orbit_direction": props.get("sat:orbit_state")
        or props.get("sar:orbit_state"),
        "relative_orbit": props.get("sat:relative_orbit"),
        "platform": props.get("platform"),
        "polarizations": props.get("sar:polarizations"),
        "bbox": list(item.bbox) if item.bbox else None,
        "assets_used": ",".join(assets_used),
    }


def build_provenance_table(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Consolida registros de proveniência em um DataFrame (salvável em Parquet)."""
    df = pd.DataFrame(records)
    logger.info("Tabela de proveniência: %d linhas, %d colunas.", len(df), df.shape[1])
    return df


def filter_s2_by_cloud(items: List[Any], max_cloud: float) -> List[Any]:
    """Pré-filtro grosseiro de S2 por ``eo:cloud_cover`` do tile inteiro.

    ATENÇÃO metodológica: este é apenas um pré-filtro barato. O controle de
    qualidade fino (fração de nuvens DENTRO da AOI via SCL) é feito depois, em
    ``sentinel2_preprocessing``. Não confie só neste número (ETAPA 6).
    """
    kept = [it for it in items if (it.properties.get("eo:cloud_cover") or 100) <= max_cloud]
    logger.info("Pré-filtro S2 (cloud<=%.0f%%): %d -> %d items.",
                max_cloud, len(items), len(kept))
    return kept


def dominant_orbit_direction(items: List[Any]) -> str | None:
    """Determina a direção de órbita SAR mais frequente entre os Items.

    Usado para a decisão da ETAPA 11 (usar uma única geometria dominante na PoC).
    """
    counts: Dict[str, int] = {}
    for it in items:
        d = it.properties.get("sat:orbit_state") or it.properties.get("sar:orbit_state")
        if d:
            counts[d] = counts.get(d, 0) + 1
    if not counts:
        return None
    dom = max(counts, key=counts.get)
    logger.info("Órbita SAR dominante: %s (%s).", dom, counts)
    return dom
