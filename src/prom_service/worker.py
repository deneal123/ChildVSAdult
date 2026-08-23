from __future__ import annotations

import logging
import time

from .artifacts import load_artifacts
from .config import Settings
from .db import Database
from .image import InferenceEngine, MediaFetcher
from .service import PromService

log = logging.getLogger(__name__)


def build_worker(settings: Settings) -> PromService:
    artifacts = load_artifacts(settings.model_manifest)
    return PromService(
        db=Database(settings.database_url),
        artifacts=artifacts,
        fetcher=MediaFetcher(settings),
        engine=InferenceEngine(artifacts),
    )


def run() -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = Settings.from_env()
    service = build_worker(settings)
    log.info("worker started model_version=%s", service.artifacts.version)
    while True:
        if not service.process_one_job():
            time.sleep(settings.worker_poll_seconds)
