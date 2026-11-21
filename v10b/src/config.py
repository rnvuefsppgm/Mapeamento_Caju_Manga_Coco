"""Carregamento de configuração e utilitários de logging.

Centraliza o acesso ao ``config.yaml`` para que nenhum script precise de caminhos
hardcoded. Fornece também ``setup_logging`` para logs consistentes em todos os
scripts (arquivo + console).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

# Raiz do projeto = pasta que contém a pasta ``config``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Lê o arquivo YAML de configuração e devolve um dicionário.

    Args:
        config_path: caminho do YAML. Se ``None``, usa ``config/config.yaml``.

    Returns:
        Dicionário de configuração.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # Raiz do código (onde vivem os scripts e o config) — imutável.
    cfg["_project_root"] = str(PROJECT_ROOT)

    # Raiz de SAÍDA — onde todas as pastas de dados/resultados serão criadas.
    # Se ``output_root`` estiver definido no YAML (caminho absoluto, ex.:
    # 'C:\\Users\\Rodrigo\\Desktop\\...\\Modelo'), TODAS as saídas vão para lá.
    # Se vazio/ausente, mantém o comportamento antigo (relativo ao projeto).
    out_root = cfg.get("output_root")
    cfg["_output_root"] = str(Path(out_root).expanduser()) if out_root else str(PROJECT_ROOT)
    return cfg


def output_root(cfg: Dict[str, Any]) -> Path:
    """Retorna a raiz de saída (absoluta) onde os resultados são gravados."""
    return Path(cfg["_output_root"]).resolve()


def resolve_path(cfg: Dict[str, Any], key: str) -> Path:
    """Resolve um caminho de ``cfg['paths']`` para caminho absoluto.

    Regras:
        * se o valor da chave for um caminho ABSOLUTO, ele é usado como está
          (permite, por exemplo, apontar ``vectors`` para uma pasta de entrada
          fora da raiz de saída);
        * caso contrário, o caminho relativo é resolvido sob ``output_root``
          (definido em ``config.yaml``), e não mais sob a raiz do código.
    """
    rel = Path(cfg["paths"][key])
    if rel.is_absolute():
        return rel.resolve()
    return (output_root(cfg) / rel).resolve()


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    """Garante que todas as pastas declaradas em ``paths`` existam."""
    for key in cfg["paths"]:
        resolve_path(cfg, key).mkdir(parents=True, exist_ok=True)


def setup_logging(name: str, cfg: Dict[str, Any] | None = None,
                  level: int = logging.INFO) -> logging.Logger:
    """Configura logging para console e arquivo (em ``paths.logs``).

    Args:
        name: nome do logger (normalmente o nome do script).
        cfg: configuração (para localizar a pasta de logs). Opcional.
        level: nível de logging.

    Returns:
        Logger configurado.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if cfg is not None:
        log_dir = resolve_path(cfg, "logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
