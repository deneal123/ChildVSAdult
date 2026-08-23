from __future__ import annotations

import pytest
from PIL import Image

from prom_service.image import MediaError, assess_quality, validate_media_url


def test_media_url_rejects_ssrf_shapes(settings):
    validate_media_url("https://media.internal.example/photo.jpg", settings.allowed_media_hosts)
    for url in [
        "http://media.internal.example/photo.jpg",
        "https://user@media.internal.example/photo.jpg",
        "https://localhost/photo.jpg",
        "https://media.internal.example:8443/photo.jpg",
    ]:
        with pytest.raises(MediaError):
            validate_media_url(url, settings.allowed_media_hosts)


def test_blank_or_small_images_fail_quality(settings):
    with pytest.raises(MediaError):
        assess_quality(Image.new("RGB", (100, 100)), settings)
    with pytest.raises(MediaError):
        assess_quality(Image.new("RGB", (256, 256)), settings)
