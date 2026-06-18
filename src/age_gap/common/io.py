"""Чтение/запись JSONL и разрешение путей проекта.

Каждый этап конвейера пишет и читает свои сущности построчным JSON (JSONL): это устойчиво
к частичным сбоям, удобно для дозаписи и стримингового чтения больших коллекций.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from age_gap.settings import paths_settings

# Корень репозитория: .../src/age_gap/common/io.py -> подняться на 3 уровня.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(*parts: str) -> Path:
    """Абсолютный путь относительно корня проекта."""
    return PROJECT_ROOT.joinpath(*parts)


def data_path(key: str, *extra: str) -> Path:
    """Путь из секции [default.paths] settings.toml по ключу (например, "data_dir")."""
    base = str(getattr(paths_settings(), key))
    return resolve_path(base, *extra)


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> int:
    """Записать (перезаписать) коллекцию dict-ов в JSONL. Возвращает число строк."""
    path = Path(path)
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
            count += 1
    return count


def append_jsonl(path: Path | str, row: dict[str, Any]) -> None:
    path = Path(path)
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False))
        f.write("\n")


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Лениво прочитать JSONL. Пустые строки пропускаются."""
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
