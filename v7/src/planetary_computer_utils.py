"""Conexão com o catálogo STAC do Microsoft Planetary Computer e assinatura SAS.

Referências oficiais:
    - SAS / assinatura de assets: https://planetarycomputer.microsoft.com/docs/concepts/sas/
    - STAC API:                   https://planetarycomputer.microsoft.com/docs/reference/stac/

Notas validadas na documentação (ago/2026):
    * Endpoint STAC público: ``https://planetarycomputer.microsoft.com/api/stac/v1``.
    * A assinatura SAS é feita com ``planetary_computer.sign`` (ou ``sign_inplace``).
    * O acesso ao ``sentinel-1-rtc`` deixou de exigir API key; tokens SAS anônimos
      já acessam os dados. Uma subscription key (env ``PC_SDK_SUBSCRIPTION_KEY``)
      ainda é suportada e eleva limites de rate.
    * O catálogo público está em "Preview" e a Microsoft direciona novos usuários
      ao Planetary Computer Pro (autenticação Entra, SDK ``azure-planetarycomputer``).
      Este pipeline usa a API pública; migrar para o Pro exigiria trocar a camada
      de autenticação (ver README, seção "Solução de problemas").
"""
from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, TypeVar

import planetary_computer as pc
import pystac_client
from pystac_client import Client

logger = logging.getLogger(__name__)

T = TypeVar("T")


def open_catalog(cfg: Dict[str, Any]) -> Client:
    """Abre o catálogo STAC do Planetary Computer com assinatura SAS automática.

    A ``modifier=pc.sign_inplace`` faz com que cada Item/Asset devolvido pela busca
    já venha com URL assinada (SAS), evitando 403 na leitura remota dos COGs.

    Args:
        cfg: configuração do projeto.

    Returns:
        Cliente ``pystac_client.Client`` conectado ao endpoint.
    """
    endpoint = cfg["stac"]["endpoint"]
    modifier = pc.sign_inplace if cfg["stac"].get("sign_assets", True) else None
    logger.info("Abrindo catálogo STAC: %s (assinatura SAS=%s)",
                endpoint, modifier is not None)
    catalog = pystac_client.Client.open(endpoint, modifier=modifier)
    return catalog


def sign_url(url: str) -> str:
    """Assina uma URL isolada de asset com um token SAS.

    Útil quando se possui apenas o href e não o Item completo.
    """
    return pc.sign(url)


def with_retry(
    max_retries: int = 5,
    backoff_base: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorador de *retry* com *exponential backoff*.

    Envolve chamadas de rede (consultas STAC, leitura remota) para torná-las
    resilientes a falhas transitórias (timeouts, 429, 5xx).

    Args:
        max_retries: número máximo de tentativas.
        backoff_base: base do backoff exponencial (segundos).
        exceptions: exceções que disparam nova tentativa.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: BaseException | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # noqa: BLE001
                    last_exc = exc
                    wait = backoff_base ** attempt
                    logger.warning(
                        "Tentativa %d/%d falhou em %s: %s. Aguardando %.1fs.",
                        attempt, max_retries, func.__name__, exc, wait,
                    )
                    time.sleep(wait)
            assert last_exc is not None
            raise last_exc
        return wrapper
    return decorator


def discover_sentinel_collections(catalog: Client) -> Dict[str, list[str]]:
    """Lista coleções do catálogo que contêm 'sentinel-1'/'sentinel-2' no ID.

    Serve para *validar em tempo de execução* os IDs de coleção antes de usá-los,
    conforme exigido pela metodologia (não assumir IDs de versões antigas).

    Returns:
        Dicionário {"sentinel-1": [...ids...], "sentinel-2": [...ids...]}.
    """
    found: Dict[str, list[str]] = {"sentinel-1": [], "sentinel-2": []}
    for coll in catalog.get_collections():
        cid = coll.id
        if "sentinel-1" in cid:
            found["sentinel-1"].append(cid)
        elif "sentinel-2" in cid:
            found["sentinel-2"].append(cid)
    logger.info("Coleções Sentinel disponíveis: %s", found)
    return found
