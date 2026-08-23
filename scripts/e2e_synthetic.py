"""Production-image E2E contract test using synthetic pixels and a disposable PostgreSQL database."""

from __future__ import annotations

import io

import httpx
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from prom_service.app import create_app
from prom_service.artifacts import load_artifacts
from prom_service.config import Settings
from prom_service.db import Database
from prom_service.image import InferenceEngine, MediaFetcher
from prom_service.service import PromService

AUTH = {"Authorization": "Bearer e2e-internal-token"}
URL = "https://media.e2e.invalid/synthetic.jpg"


def synthetic_jpeg() -> bytes:
    rows, columns = np.indices((256, 256))
    pixels = np.stack(
        [(rows * 13 + columns * 7) % 256, (rows * 3 + columns * 17) % 256, (rows * 19 + columns * 5) % 256],
        axis=-1,
    ).astype(np.uint8)
    image = Image.fromarray(pixels)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=95)
    return output.getvalue()


def main() -> None:
    settings = Settings.from_env()
    database = Database(settings.database_url)
    database.apply_migrations()
    payload = synthetic_jpeg()

    def fetch(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == URL
        return httpx.Response(200, headers={"content-type": "image/jpeg", "content-length": str(len(payload))}, content=payload)

    artifacts = load_artifacts(settings.model_manifest)
    service = PromService(
        db=database,
        artifacts=artifacts,
        fetcher=MediaFetcher(settings, httpx.Client(transport=httpx.MockTransport(fetch))),
        engine=InferenceEngine(artifacts),
    )
    app = create_app(settings, service)
    with TestClient(app) as client:
        assert client.get("/live").status_code == 200
        assert client.get("/ready").status_code == 200
        assert client.post("/v1/rank", json={}).status_code == 401
        for profile_id, account_id in (("candidate-a", "account-a"), ("candidate-b", "account-b")):
            queued = client.put(
                f"/v1/profiles/{profile_id}/representation",
                headers=AUTH,
                json={"account_id": account_id, "photo_version": "v1", "image_url": URL},
            )
            assert queued.status_code == 202, queued.text
        assert service.process_one_job() and service.process_one_job()
        for profile_id in ("candidate-a", "candidate-b"):
            assert client.get(f"/v1/profiles/{profile_id}/representation", headers=AUTH).json()["status"] == "ready"
        neutral = client.post(
            "/v1/rank",
            headers=AUTH,
            json={"viewer_id": "viewer", "request_id": "cold", "candidate_ids": ["candidate-a", "missing", "candidate-b"]},
        )
        assert neutral.status_code == 200 and neutral.json()["mode"] == "neutral"
        assert neutral.json()["unavailable_candidate_ids"] == ["missing"]
        swipe = client.post(
            "/v1/swipes",
            headers=AUTH,
            json={"event_id": "e2e-like-1", "viewer_id": "viewer", "candidate_id": "candidate-a", "reaction": "like", "impression_id": "cold"},
        )
        assert swipe.json() == {"applied": True, "personalization_updated": True}
        duplicate = client.post(
            "/v1/profiles/candidate-a/duplicate-check",
            headers=AUTH,
            json={"account_id": "account-a", "image_url": URL, "biometric_consent": True},
        )
        assert duplicate.status_code == 200 and duplicate.json()["comparison_scope"] == "same_profile_only"
        assert duplicate.json()["is_likely_duplicate"] is True
        assert client.post(
            "/v1/profiles/candidate-a/duplicate-check",
            headers=AUTH,
            json={"account_id": "account-b", "image_url": URL, "biometric_consent": True},
        ).status_code == 404
        personalized = client.post(
            "/v1/rank",
            headers=AUTH,
            json={"viewer_id": "viewer", "request_id": "warm", "candidate_ids": ["candidate-a", "candidate-b"]},
        )
        assert personalized.status_code == 200 and personalized.json()["mode"] == "personalized"
        assert client.delete("/v1/accounts/account-a", headers=AUTH).json()["erased_profile_representations"] == 1
        assert client.get("/v1/profiles/candidate-a/representation", headers=AUTH).status_code == 404
    print('{"status":"ok","scenario":"synthetic-api-worker-postgres-onnx"}')


if __name__ == "__main__":
    main()
