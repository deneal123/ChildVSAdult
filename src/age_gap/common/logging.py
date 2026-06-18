"""Единая настройка логирования.

Уровень и файл берутся из секции [default.logs] settings.toml. Логи пишутся и в консоль,
и в файл (например, причины реджекта лиц на этапе препроцессинга — SKILL §11).
"""

from __future__ import annotations

import logging
from pathlib import Path

from age_gap.settings import settings

_CONFIGURED = False


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def setup_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = str(getattr(settings.logs, "level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    log_file = getattr(settings.logs, "file", None)
    if log_file:
        path = _project_root() / str(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=handlers,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
