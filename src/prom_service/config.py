from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    database_url: str
    internal_token: str
    model_manifest: Path
    allowed_media_hosts: frozenset[str]
    media_max_bytes: int
    media_timeout_seconds: float
    min_image_side: int
    min_blur_variance: float
    worker_poll_seconds: float
    worker_lease_seconds: int
    worker_max_attempts: int
    service_version: str

    @classmethod
    def from_env(cls) -> Settings:
        def required(name: str) -> str:
            value = os.getenv(name, "").strip()
            if not value or value.startswith("replace-"):
                raise ConfigurationError(f"{name} must be set from the secret/configuration store")
            return value

        hosts = frozenset(h.strip().lower() for h in required("PROM_ALLOWED_MEDIA_HOSTS").split(",") if h.strip())
        if not hosts:
            raise ConfigurationError("PROM_ALLOWED_MEDIA_HOSTS must contain at least one host")
        return cls(
            database_url=required("DATABASE_URL"),
            internal_token=required("PROM_INTERNAL_TOKEN"),
            model_manifest=Path(required("PROM_MODEL_MANIFEST")),
            allowed_media_hosts=hosts,
            media_max_bytes=int(os.getenv("PROM_MEDIA_MAX_BYTES", "8388608")),
            media_timeout_seconds=float(os.getenv("PROM_MEDIA_TIMEOUT_SECONDS", "5")),
            min_image_side=int(os.getenv("PROM_MIN_IMAGE_SIDE", "160")),
            min_blur_variance=float(os.getenv("PROM_MIN_BLUR_VARIANCE", "12")),
            worker_poll_seconds=float(os.getenv("PROM_WORKER_POLL_SECONDS", "1")),
            worker_lease_seconds=int(os.getenv("PROM_WORKER_LEASE_SECONDS", "60")),
            worker_max_attempts=int(os.getenv("PROM_WORKER_MAX_ATTEMPTS", "3")),
            service_version=os.getenv("PROM_SERVICE_VERSION", "1.0.0"),
        )
