from __future__ import annotations

import hmac
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status

from . import __version__
from .artifacts import ArtifactError, load_artifacts
from .config import ConfigurationError, Settings
from .db import Database, OwnershipConflict
from .image import InferenceEngine, MediaError, MediaFetcher, validate_media_url
from .schemas import (
    DuplicateCheckRequest,
    DuplicateCheckResponse,
    EraseResponse,
    RankedCandidate,
    RankRequest,
    RankResponse,
    RepresentationStatus,
    RepresentationUpsert,
    SwipeRequest,
    SwipeResponse,
)
from .service import PromService


def _auth(request: Request, authorization: Annotated[str | None, Header()] = None) -> None:
    settings: Settings = request.app.state.settings
    expected = f"Bearer {settings.internal_token}"
    if authorization is None or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="internal authentication required")


def _service(request: Request) -> PromService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="model artifacts are unavailable")
    return service


def create_app(settings: Settings | None = None, service: PromService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is not None:
            app.state.service = service
            yield
            return
        try:
            runtime_settings: Settings = app.state.settings
            artifacts = load_artifacts(runtime_settings.model_manifest)
            app.state.service = PromService(
                db=Database(runtime_settings.database_url),
                artifacts=artifacts,
                fetcher=MediaFetcher(runtime_settings),
                engine=InferenceEngine(artifacts),
            )
            app.state.startup_error = None
        except (ArtifactError, ConfigurationError, RuntimeError) as exc:
            app.state.service = None
            app.state.startup_error = type(exc).__name__
        yield

    if settings is None:
        # Importing the ASGI application must not read secrets or download models.
        settings = Settings(
            database_url="",
            internal_token="",
            model_manifest=Path("/nonexistent"),
            allowed_media_hosts=frozenset(),
            media_max_bytes=0,
            media_timeout_seconds=0,
            min_image_side=0,
            min_blur_variance=0,
            worker_poll_seconds=1,
            worker_lease_seconds=60,
            worker_max_attempts=3,
            service_version=__version__,
        )
    app = FastAPI(title="prom internal API", version=settings.service_version, lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.settings = settings
    app.state.service = service
    app.state.startup_error = None

    @app.get("/live")
    def live() -> dict[str, str]:
        return {"status": "ok", "version": app.state.settings.service_version}

    @app.get("/version")
    def version() -> dict[str, str | bool]:
        service = app.state.service
        return {
            "service_version": app.state.settings.service_version,
            "model_version": service.artifacts.version if service else "unavailable",
            "personalization_enabled": bool(service and service.artifacts.projection),
        }

    @app.get("/ready")
    def ready() -> dict[str, str]:
        if app.state.service is None:
            raise HTTPException(status_code=503, detail="service dependencies unavailable")
        try:
            with app.state.service.db.engine.connect() as connection:
                connection.exec_driver_sql("SELECT 1")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail="database unavailable") from exc
        return {"status": "ready", "model_version": app.state.service.artifacts.version}

    @app.put("/v1/profiles/{profile_id}/representation", response_model=RepresentationStatus, status_code=202, dependencies=[Depends(_auth)])
    def queue_representation(profile_id: str, body: RepresentationUpsert, prom: PromService = Depends(_service)) -> RepresentationStatus:
        try:
            validate_media_url(str(body.image_url), prom.fetcher.settings.allowed_media_hosts)
            prom.db.queue_representation(profile_id, body.account_id, body.photo_version, str(body.image_url))
        except (MediaError, OwnershipConflict) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return RepresentationStatus(profile_id=profile_id, photo_version=body.photo_version, status="pending")

    @app.get("/v1/profiles/{profile_id}/representation", response_model=RepresentationStatus, dependencies=[Depends(_auth)])
    def representation_status(profile_id: str, prom: PromService = Depends(_service)) -> RepresentationStatus:
        item = prom.db.get_representation(profile_id)
        if item is None:
            raise HTTPException(status_code=404, detail="representation not found")
        return RepresentationStatus(profile_id=item.profile_id, photo_version=item.photo_version, status=item.status, error_code=item.error_code)

    @app.post("/v1/rank", response_model=RankResponse, dependencies=[Depends(_auth)])
    def rank(body: RankRequest, prom: PromService = Depends(_service)) -> RankResponse:
        candidates, unavailable, mode = prom.rank(body.viewer_id, body.request_id, body.candidate_ids, body.limit)
        return RankResponse(
            candidates=[RankedCandidate(profile_id=profile_id, score=score) for profile_id, score in candidates],
            unavailable_candidate_ids=unavailable,
            mode=mode,
        )

    @app.post("/v1/swipes", response_model=SwipeResponse, dependencies=[Depends(_auth)])
    def swipe(body: SwipeRequest, prom: PromService = Depends(_service)) -> SwipeResponse:
        applied, updated = prom.swipe(body.event_id, body.viewer_id, body.candidate_id, body.reaction, body.impression_id)
        return SwipeResponse(applied=applied, personalization_updated=updated)

    @app.post("/v1/profiles/{profile_id}/duplicate-check", response_model=DuplicateCheckResponse, dependencies=[Depends(_auth)])
    def duplicate_check(profile_id: str, body: DuplicateCheckRequest, prom: PromService = Depends(_service)) -> DuplicateCheckResponse:
        try:
            similarity, is_duplicate = prom.duplicate_check(profile_id, body.account_id, str(body.image_url))
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except MediaError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return DuplicateCheckResponse(
            profile_id=profile_id,
            comparison_scope="same_profile_only",
            similarity=similarity,
            is_likely_duplicate=is_duplicate,
        )

    @app.delete("/v1/accounts/{account_id}", response_model=EraseResponse, dependencies=[Depends(_auth)])
    def erase_account(account_id: str, prom: PromService = Depends(_service)) -> EraseResponse:
        return EraseResponse(erased_profile_representations=prom.db.erase_account(account_id))

    return app


app = create_app()


def run() -> None:
    import uvicorn

    runtime_settings = Settings.from_env()
    uvicorn.run(create_app(runtime_settings), host="0.0.0.0", port=8080, proxy_headers=False)
