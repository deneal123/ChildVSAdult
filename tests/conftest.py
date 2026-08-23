from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from prom_service.artifacts import ModelArtifacts, ModelSpec, Projection
from prom_service.config import Settings
from prom_service.db import Database
from prom_service.service import PromService


def _image() -> Image.Image:
    pixels = np.indices((256, 256)).sum(axis=0) % 2 * 255
    return Image.fromarray(np.stack([pixels, pixels, pixels], axis=-1).astype(np.uint8))


@dataclass
class FakeFetcher:
    settings: Settings

    def fetch(self, url: str) -> Image.Image:
        return _image()


class FakeEngine:
    def representations(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        return np.array([1.0, 0.0, 0.0]), np.array([0.9, 0.1, 0.0])


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'prom.db'}",
        internal_token="test-token",
        model_manifest=tmp_path / "manifest.json",
        allowed_media_hosts=frozenset({"media.internal.example"}),
        media_max_bytes=1024 * 1024,
        media_timeout_seconds=1,
        min_image_side=160,
        min_blur_variance=12,
        worker_poll_seconds=0.01,
        service_version="test",
    )


@pytest.fixture
def artifacts(tmp_path: Path) -> ModelArtifacts:
    spec = ModelSpec(
        path=tmp_path / "unused.onnx",
        sha256="0" * 64,
        input_name="input",
        output_name="output",
        image_size=224,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )
    return ModelArtifacts(
        version="test-model-v1",
        vision=spec,
        face=spec,
        projection=Projection(
            mean=np.zeros(3), components=np.eye(3), scale=np.ones(3), version="projection-v1"
        ),
    )


@pytest.fixture
def service(settings: Settings, artifacts: ModelArtifacts) -> PromService:
    database = Database(settings.database_url)
    database.create_schema()
    return PromService(database, artifacts, FakeFetcher(settings), FakeEngine())  # type: ignore[arg-type]
