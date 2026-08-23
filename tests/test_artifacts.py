from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from prom_service.artifacts import ArtifactError, load_artifacts


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_requires_matching_checksums(tmp_path):
    vision = tmp_path / "vision.onnx"
    face = tmp_path / "face.onnx"
    projection = tmp_path / "projection.npz"
    vision.write_bytes(b"vision")
    face.write_bytes(b"face")
    np.savez(projection, mean=np.zeros(3), components=np.eye(2, 3), scale=np.ones(3))
    spec = {
        "input": "input",
        "output": "output",
        "image_size": 224,
        "mean": [0, 0, 0],
        "std": [1, 1, 1],
        "color_order": "rgb",
    }
    manifest = {
        "version": "v1",
        "vision": {**spec, "path": vision.name, "sha256": _digest(vision)},
        "face": {**spec, "path": face.name, "sha256": _digest(face)},
        "projection": {"path": projection.name, "sha256": _digest(projection), "version": "p1"},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = load_artifacts(path)
    assert loaded.projection is not None and loaded.projection.components.shape == (2, 3)

    vision.write_bytes(b"changed")
    with pytest.raises(ArtifactError, match="checksum"):
        load_artifacts(path)
