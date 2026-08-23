from __future__ import annotations

from .config import Settings
from .db import Database


def main() -> None:
    settings = Settings.from_env()
    applied = Database(settings.database_url).apply_migrations()
    if applied:
        print(f"applied prom migrations: {', '.join(applied)}")
    else:
        print("prom schema is current")
