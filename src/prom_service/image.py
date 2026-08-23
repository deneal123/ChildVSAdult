from __future__ import annotations

import io
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .artifacts import ModelArtifacts, ModelSpec
from .config import Settings


class MediaError(ValueError):
    """A safe, client-facing media validation failure."""


@dataclass(frozen=True)
class Quality:
    width: int
    height: int
    blur_variance: float


def validate_media_url(url: str, allowed_hosts: frozenset[str]) -> None:
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        raise MediaError("media URL must be an HTTPS URL without credentials")
    if parts.hostname.lower() not in allowed_hosts:
        raise MediaError("media URL host is not approved")
    if parts.port not in (None, 443):
        raise MediaError("media URL port is not approved")


class MediaFetcher:
    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.settings = settings
        self.client = client or httpx.Client(
            follow_redirects=False,
            timeout=settings.media_timeout_seconds,
            headers={"Accept": "image/jpeg,image/png,image/webp"},
        )

    def fetch(self, url: str) -> Image.Image:
        validate_media_url(url, self.settings.allowed_media_hosts)
        try:
            with self.client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise MediaError("approved media could not be retrieved")
                content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                if content_type not in {"image/jpeg", "image/png", "image/webp"}:
                    raise MediaError("approved media has an unsupported content type")
                declared = response.headers.get("content-length")
                if declared and int(declared) > self.settings.media_max_bytes:
                    raise MediaError("approved media exceeds the size limit")
                body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) > self.settings.media_max_bytes:
                        raise MediaError("approved media exceeds the size limit")
        except (httpx.HTTPError, ValueError) as exc:
            if isinstance(exc, MediaError):
                raise
            raise MediaError("approved media could not be retrieved") from exc
        try:
            image = Image.open(io.BytesIO(body))
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise MediaError("approved media is not a valid image") from exc


def assess_quality(image: Image.Image, settings: Settings) -> Quality:
    width, height = image.size
    if min(width, height) < settings.min_image_side:
        raise MediaError("image is smaller than the approved minimum")
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    # Variance of a discrete Laplacian catches uniformly blurred/blank uploads without retaining pixels.
    laplacian = -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    variance = float(np.var(laplacian))
    if not np.isfinite(variance) or variance < settings.min_blur_variance:
        raise MediaError("image quality is insufficient")
    return Quality(width=width, height=height, blur_variance=variance)


class InferenceEngine:
    """ONNX-only inference. Artifact retrieval is deliberately outside the request path."""

    def __init__(self, artifacts: ModelArtifacts):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - dependency failure is deployment-specific
            raise RuntimeError("onnxruntime is required for inference") from exc
        self.artifacts = artifacts
        self._vision = ort.InferenceSession(str(artifacts.vision.path), providers=["CPUExecutionProvider"])
        self._face = ort.InferenceSession(str(artifacts.face.path), providers=["CPUExecutionProvider"])

    @staticmethod
    def _input(image: Image.Image, spec: ModelSpec) -> np.ndarray:
        pixels = np.asarray(image.resize((spec.image_size, spec.image_size)), dtype=np.float32) / 255.0
        if spec.color_order == "bgr":
            pixels = pixels[..., ::-1]
        normalized = (pixels - np.asarray(spec.mean, dtype=np.float32)) / np.asarray(spec.std, dtype=np.float32)
        return np.ascontiguousarray(np.transpose(normalized, (2, 0, 1))[None, ...])

    @staticmethod
    def _normalise(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(vector).all() or norm <= 1e-12:
            raise MediaError("model returned an invalid image representation")
        return vector / norm

    def _embed(self, session: object, image: Image.Image, spec: ModelSpec) -> np.ndarray:
        output = session.run([spec.output_name], {spec.input_name: self._input(image, spec)})[0]
        return self._normalise(output)

    def representations(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._embed(self._vision, image, self.artifacts.vision),
            self._embed(self._face, image, self.artifacts.face),
        )
