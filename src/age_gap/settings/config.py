from pathlib import Path

from dotenv import load_dotenv
from dynaconf import Dynaconf

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

settings = Dynaconf(
    environments=True,
    envvar_prefix=False,
    settings_files=[str(BASE_DIR / "settings.toml")],
)


def base_settings():
    """Базовые настройки проекта (секция [default.base])."""
    return getattr(settings.base, "base", settings.base)


def paths_settings():
    """Пути проекта (секция [default.paths])."""
    return settings.paths
