from __future__ import annotations

from .config import Settings
from .db import Database


def main() -> None:
    settings = Settings.from_env()
    Database(settings.database_url).create_schema()
    print("prom schema is current")
