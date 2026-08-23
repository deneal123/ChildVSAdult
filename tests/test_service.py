from __future__ import annotations

from fastapi.testclient import TestClient

from prom_service.app import create_app

AUTH = {"Authorization": "Bearer test-token"}
URL = "https://media.internal.example/profile.jpg"


def _ready(service, profile_id: str, account_id: str, photo_version: str = "v1"):
    service.db.queue_representation(profile_id, account_id, photo_version, URL)
    assert service.process_one_job()


def test_internal_api_auth_and_queue(service, settings):
    app = create_app(settings, service)
    with TestClient(app) as client:
        assert client.get("/version").json()["model_version"] == "test-model-v1"
        assert client.put("/v1/profiles/p1/representation", json={}).status_code == 401
        response = client.put(
            "/v1/profiles/p1/representation",
            headers=AUTH,
            json={"account_id": "a1", "photo_version": "v1", "image_url": URL},
        )
        assert response.status_code == 202
        assert response.json()["status"] == "pending"
        assert service.process_one_job()
        status = client.get("/v1/profiles/p1/representation", headers=AUTH)
        assert status.json()["status"] == "ready"


def test_cold_start_then_idempotent_personalisation(service, settings):
    _ready(service, "candidate-a", "account-a")
    _ready(service, "candidate-b", "account-b")
    app = create_app(settings, service)
    with TestClient(app) as client:
        first = client.post(
            "/v1/rank",
            headers=AUTH,
            json={"viewer_id": "viewer", "request_id": "r1", "candidate_ids": ["candidate-a", "missing", "candidate-b"]},
        )
        assert first.status_code == 200
        assert first.json()["mode"] == "neutral"
        assert first.json()["unavailable_candidate_ids"] == ["missing"]
        swipe = {"event_id": "e1", "viewer_id": "viewer", "candidate_id": "candidate-a", "reaction": "like", "impression_id": "i1"}
        assert client.post("/v1/swipes", headers=AUTH, json=swipe).json() == {"applied": True, "personalization_updated": True}
        assert client.post("/v1/swipes", headers=AUTH, json=swipe).json() == {"applied": False, "personalization_updated": False}
        ranked = client.post(
            "/v1/rank",
            headers=AUTH,
            json={"viewer_id": "viewer", "request_id": "r2", "candidate_ids": ["candidate-a", "candidate-b"]},
        )
        assert ranked.json()["mode"] == "personalized"
        assert ranked.json()["candidates"][0]["profile_id"] == "candidate-a"


def test_duplicate_scope_and_erasure(service, settings):
    _ready(service, "p1", "account-1")
    app = create_app(settings, service)
    with TestClient(app) as client:
        denied = client.post(
            "/v1/profiles/p1/duplicate-check",
            headers=AUTH,
            json={"account_id": "other-account", "image_url": URL, "biometric_consent": True},
        )
        assert denied.status_code == 404
        checked = client.post(
            "/v1/profiles/p1/duplicate-check",
            headers=AUTH,
            json={"account_id": "account-1", "image_url": URL, "biometric_consent": True},
        )
        assert checked.json()["comparison_scope"] == "same_profile_only"
        assert checked.json()["is_likely_duplicate"] is True
        erased = client.delete("/v1/accounts/account-1", headers=AUTH)
        assert erased.json()["erased_profile_representations"] == 1
        assert service.db.get_representation("p1") is None
