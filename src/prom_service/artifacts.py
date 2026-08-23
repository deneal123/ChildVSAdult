from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class ArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    path: Path
    sha256: str
    input_name: str
    output_name: str
    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    color_order: str


@dataclass(frozen=True)
class Projection:
    mean: np.ndarray
    components: np.ndarray
    scale: np.ndarray
    version: str

    def transform(self, vector: np.ndarray) -> np.ndarray:
        if vector.shape != self.mean.shape:
            raise ArtifactError("vision embedding dimension does not match the projection artifact")
        projected = ((vector - self.mean) / self.scale) @ self.components.T
        if not np.isfinite(projected).all():
            raise ArtifactError("projection produced a non-finite vector")
        return projected.astype(np.float64)


@dataclass(frozen=True)
class ModelArtifacts:
    version: str
    vision: ModelSpec
    face: ModelSpec
    projection: Projection | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ArtifactError(f"{label}.path is required")
    path = (root / value).resolve()
    if root.resolve() not in path.parents:
        raise ArtifactError(f"{label}.path must stay inside the model-artifact directory")
    return path


def _spec(root: Path, payload: Any, label: str) -> ModelSpec:
    if not isinstance(payload, dict):
        raise ArtifactError(f"{label} specification is required")
    path = _resolve(root, payload.get("path"), label)
    expected = payload.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ArtifactError(f"{label}.sha256 must be a SHA-256 hex digest")
    if not path.is_file() or _sha256(path).lower() != expected.lower():
        raise ArtifactError(f"{label} artifact is missing or its checksum does not match")
    try:
        mean = tuple(float(x) for x in payload["mean"])
        std = tuple(float(x) for x in payload["std"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(f"{label} normalization parameters are invalid") from exc
    if len(mean) != 3 or len(std) != 3 or any(v <= 0 for v in std):
        raise ArtifactError(f"{label} normalization must contain three positive standard deviations")
    image_size = int(payload.get("image_size", 0))
    if image_size < 32:
        raise ArtifactError(f"{label}.image_size must be at least 32")
    color_order = str(payload.get("color_order", "rgb")).lower()
    if color_order not in {"rgb", "bgr"}:
        raise ArtifactError(f"{label}.color_order must be rgb or bgr")
    return ModelSpec(
        path=path,
        sha256=expected.lower(),
        input_name=str(payload.get("input", "input")),
        output_name=str(payload.get("output", "output")),
        image_size=image_size,
        mean=mean,
        std=std,
        color_order=color_order,
    )


def _projection(root: Path, payload: Any) -> Projection | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ArtifactError("projection specification is invalid")
    path = _resolve(root, payload.get("path"), "projection")
    expected = payload.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64 or not path.is_file():
        raise ArtifactError("projection artifact is missing")
    if _sha256(path).lower() != expected.lower():
        raise ArtifactError("projection checksum does not match")
    try:
        data = np.load(path, allow_pickle=False)
        mean = np.asarray(data["mean"], dtype=np.float64)
        components = np.asarray(data["components"], dtype=np.float64)
        scale = np.asarray(data["scale"], dtype=np.float64)
    except Exception as exc:  # np.load emits multiple exception classes
        raise ArtifactError("projection artifact must contain mean, components and scale arrays") from exc
    if mean.ndim != 1 or scale.shape != mean.shape or components.ndim != 2 or components.shape[1] != len(mean):
        raise ArtifactError("projection dimensions are inconsistent")
    if not np.isfinite(mean).all() or not np.isfinite(components).all() or np.any(scale <= 0):
        raise ArtifactError("projection contains invalid numeric values")
    version = payload.get("version")
    if not isinstance(version, str) or not version:
        raise ArtifactError("projection.version is required")
    return Projection(mean=mean, components=components, scale=scale, version=version)


def load_artifacts(manifest_path: Path) -> ModelArtifacts:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError("model manifest is unreadable") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("version"), str):
        raise ArtifactError("model manifest version is required")
    root = manifest_path.resolve().parent
    return ModelArtifacts(
        version=payload["version"],
        vision=_spec(root, payload.get("vision"), "vision"),
        face=_spec(root, payload.get("face"), "face"),
        projection=_projection(root, payload.get("projection")),
    )


def main() -> None:
    from .config import Settings

    settings = Settings.from_env()
    artifacts = load_artifacts(settings.model_manifest)
    print(
        json.dumps(
            {
                "status": "ok",
                "model_version": artifacts.version,
                "projection_enabled": artifacts.projection is not None,
            }
        )
    )
