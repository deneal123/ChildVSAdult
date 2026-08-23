"""One-shot, offline-safe materialisation of the approved inference artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import load_artifacts


class ModelInitError(RuntimeError):
    pass


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ModelInitError(f"{name} is required")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(path)


def write_projection(
    path: Path, checkpoint_digest: str, input_dimensions: int, dimensions: int = 64
) -> str:
    """Create a deterministic neutral projection; it is not fitted to user data."""
    if input_dimensions < 1 or dimensions < 1 or dimensions > input_dimensions:
        raise ModelInitError("projection dimensions must be positive and not exceed its input dimensions")
    seed = int(checkpoint_digest[:16], 16)
    matrix = np.random.default_rng(seed).normal(size=(input_dimensions, dimensions))
    basis, _ = np.linalg.qr(matrix)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez(
            temporary,
            mean=np.zeros(input_dimensions, dtype=np.float32),
            components=basis.T.astype(np.float32),
            scale=np.ones(input_dimensions, dtype=np.float32),
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def _adaface_model() -> Any:
    """The published cvlface IR-50 architecture, kept only in the init image."""
    from torch import nn

    class BasicBlockIR(nn.Module):
        def __init__(self, in_channel: int, depth: int, stride: int) -> None:
            super().__init__()
            self.shortcut_layer: nn.Module
            if in_channel == depth:
                self.shortcut_layer = nn.MaxPool2d(1, stride)
            else:
                self.shortcut_layer = nn.Sequential(
                    nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False), nn.BatchNorm2d(depth)
                )
            self.res_layer = nn.Sequential(
                nn.BatchNorm2d(in_channel),
                nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
                nn.BatchNorm2d(depth),
                nn.PReLU(depth),
                nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
                nn.BatchNorm2d(depth),
            )

        def forward(self, value: Any) -> Any:
            return self.res_layer(value) + self.shortcut_layer(value)

    block = namedtuple("Block", ["in_channel", "depth", "stride"])

    def blocks(in_channel: int, depth: int, count: int) -> list[Any]:
        return [block(in_channel, depth, 2)] + [block(depth, depth, 1) for _ in range(count - 1)]

    class Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_layer = nn.Sequential(
                nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU(64)
            )
            layout = blocks(64, 64, 3) + blocks(64, 128, 4) + blocks(128, 256, 14) + blocks(256, 512, 3)
            self.body = nn.Sequential(*(BasicBlockIR(item.in_channel, item.depth, item.stride) for item in layout))
            self.output_layer = nn.Sequential(
                nn.BatchNorm2d(512), nn.Dropout(0.4), nn.Flatten(), nn.Linear(512 * 7 * 7, 512), nn.BatchNorm1d(512, affine=False)
            )

        def forward(self, value: Any) -> Any:
            return self.output_layer(self.body(self.input_layer(value)))

    class NormalizedBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = Backbone()

        def forward(self, value: Any) -> Any:
            return nn.functional.normalize(self.net(value), dim=-1)

    return NormalizedBackbone()


def _load_checkpoint(model: Any, checkpoint: Path) -> None:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ModelInitError("checkpoint does not contain a state dictionary")
    weights = {key[4:]: value for key, value in state.items() if key.startswith("net.")}
    missing, unexpected = model.net.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise ModelInitError(
            f"checkpoint architecture mismatch (missing={len(missing)}, unexpected={len(unexpected)})"
        )


def _export_face_onnx(checkpoint: Path, output: Path) -> None:
    import torch

    model = _adaface_model()
    _load_checkpoint(model, checkpoint)
    model.eval()
    example = torch.linspace(-1.0, 1.0, 3 * 112 * 112, dtype=torch.float32).reshape(1, 3, 112, 112)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".onnx", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.onnx.export(
            model,
            (example,),
            temporary,
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        import onnxruntime as ort

        expected = model(example).detach().numpy()
        actual = ort.InferenceSession(str(temporary), providers=["CPUExecutionProvider"]).run(
            ["output"], {"input": example.numpy()}
        )[0]
        if actual.shape != (1, 512) or not np.allclose(expected, actual, rtol=1e-3, atol=2e-5):
            raise ModelInitError("PyTorch and ONNX outputs do not match")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _export_vision_onnx(checkpoint: Path, config_path: Path, output: Path) -> None:
    import torch
    from torch import nn
    from transformers import CLIPVisionConfig, CLIPVisionModel

    config = CLIPVisionConfig.from_json_file(str(config_path))
    model = CLIPVisionModel(config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ModelInitError("vision checkpoint does not contain a state dictionary")
    weights = {
        key: value
        for key, value in payload.items()
        if key.startswith("vision_model.") and not key.endswith(".position_ids")
    }
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise ModelInitError(
            f"vision checkpoint architecture mismatch (missing={len(missing)}, unexpected={len(unexpected)})"
        )

    class PooledVision(nn.Module):
        def __init__(self, vision: Any) -> None:
            super().__init__()
            self.vision = vision

        def forward(self, value: Any) -> Any:
            return self.vision(pixel_values=value).pooler_output

    exported = PooledVision(model.eval())
    example = torch.linspace(-1.0, 1.0, 3 * 224 * 224, dtype=torch.float32).reshape(1, 3, 224, 224)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".onnx", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.onnx.export(
            exported,
            (example,),
            temporary,
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        import onnxruntime as ort

        expected = exported(example).detach().numpy()
        actual = ort.InferenceSession(str(temporary), providers=["CPUExecutionProvider"]).run(
            ["output"], {"input": example.numpy()}
        )[0]
        maximum_error = float(np.max(np.abs(expected - actual))) if actual.shape == expected.shape else float("inf")
        if actual.shape != (1, 768) or not np.allclose(expected, actual, rtol=1e-3, atol=1e-4):
            raise ModelInitError(f"PyTorch and ONNX vision outputs do not match (max_abs_error={maximum_error:.6g})")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest(version: str, vision_digest: str, face_digest: str, projection_digest: str) -> dict[str, Any]:
    vision = {
        "path": "clip_vit_b32.onnx",
        "sha256": vision_digest,
        "input": "input",
        "output": "output",
        "image_size": 224,
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
        "color_order": "rgb",
    }
    face = {
        "path": "adaface_ir50.onnx",
        "sha256": face_digest,
        "input": "input",
        "output": "output",
        "image_size": 112,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "color_order": "bgr",
    }
    return {
        "version": version,
        "vision": vision,
        "face": face,
        "projection": {
            "path": "projection.npz",
            "sha256": projection_digest,
            "version": "bootstrap-orthogonal-64-v1",
        },
    }


def main() -> None:
    model_dir = Path(os.environ.get("PROM_MODEL_DIR", "/app/models")).resolve()
    manifest_path = Path(os.environ.get("PROM_MODEL_MANIFEST", model_dir / "manifest.json")).resolve()
    version = _required("PROM_MODEL_VERSION")
    if manifest_path.is_file():
        loaded = load_artifacts(manifest_path)
        if loaded.version == version:
            print(json.dumps({"status": "already_ready", "model_version": version}))
            return
    source_digest = _required("PROM_MODEL_SOURCE_SHA256").lower()
    if len(source_digest) != 64 or any(char not in "0123456789abcdef" for char in source_digest):
        raise ModelInitError("PROM_MODEL_SOURCE_SHA256 must be a SHA-256 hex digest")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - only the init image installs this extra
        raise ModelInitError("huggingface-hub is required for model initialisation") from exc
    face_checkpoint = Path(
        hf_hub_download(repo_id=_required("PROM_MODEL_SOURCE_REPO"), filename=_required("PROM_MODEL_SOURCE_FILE"))
    )
    if sha256_file(face_checkpoint) != source_digest:
        raise ModelInitError("downloaded checkpoint checksum does not match PROM_MODEL_SOURCE_SHA256")
    vision_digest = _required("PROM_VISION_SOURCE_SHA256").lower()
    vision_config_digest = _required("PROM_VISION_CONFIG_SHA256").lower()
    if any(len(item) != 64 or any(char not in "0123456789abcdef" for char in item) for item in (vision_digest, vision_config_digest)):
        raise ModelInitError("vision source checksums must be SHA-256 hex digests")
    vision_repo = _required("PROM_VISION_SOURCE_REPO")
    vision_revision = _required("PROM_VISION_SOURCE_REVISION")
    vision_checkpoint = Path(
        hf_hub_download(
            repo_id=vision_repo,
            filename=_required("PROM_VISION_SOURCE_FILE"),
            revision=vision_revision,
        )
    )
    vision_config = Path(
        hf_hub_download(
            repo_id=vision_repo,
            filename=_required("PROM_VISION_CONFIG_FILE"),
            revision=vision_revision,
        )
    )
    if sha256_file(vision_checkpoint) != vision_digest or sha256_file(vision_config) != vision_config_digest:
        raise ModelInitError("downloaded vision model checksum does not match its approved configuration")
    model_dir.mkdir(parents=True, exist_ok=True)
    face_path = model_dir / "adaface_ir50.onnx"
    vision_path = model_dir / "clip_vit_b32.onnx"
    _export_face_onnx(face_checkpoint, face_path)
    _export_vision_onnx(vision_checkpoint, vision_config, vision_path)
    face_digest = sha256_file(face_path)
    exported_vision_digest = sha256_file(vision_path)
    projection_digest = write_projection(model_dir / "projection.npz", source_digest, input_dimensions=768)
    _atomic_write(
        manifest_path,
        json.dumps(_manifest(version, exported_vision_digest, face_digest, projection_digest), indent=2).encode(),
    )
    load_artifacts(manifest_path)
    print(json.dumps({"status": "ready", "model_version": version, "projection": "bootstrap-orthogonal-64-v1"}))
