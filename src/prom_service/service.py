from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .artifacts import ModelArtifacts
from .db import Database, UserModel
from .image import InferenceEngine, MediaError, MediaFetcher, assess_quality
from .recommender import BayesianState, neutral_order, project


@dataclass
class PromService:
    db: Database
    artifacts: ModelArtifacts
    fetcher: MediaFetcher
    engine: InferenceEngine

    def process_one_job(self) -> bool:
        job = self.db.claim_job(self.fetcher.settings.worker_lease_seconds)
        if job is None:
            return False
        try:
            image = self.fetcher.fetch(job.source_url)
            quality = assess_quality(image, self.fetcher.settings)
            recommendation, template = self.engine.representations(image)
            self.db.complete_job(
                job,
                recommendation.tolist(),
                template.tolist(),
                self.artifacts.version,
                {"width": quality.width, "height": quality.height, "blur_variance": quality.blur_variance},
            )
        except MediaError:
            # Do not persist image URLs or exception contents; they can contain signed query parameters.
            self.db.fail_job(
                job,
                "media_rejected",
                retryable=False,
                max_attempts=self.fetcher.settings.worker_max_attempts,
            )
        except Exception:  # noqa: BLE001 - the durable job must become observable as failed
            self.db.fail_job(
                job,
                "inference_failed",
                retryable=True,
                max_attempts=self.fetcher.settings.worker_max_attempts,
            )
        return True

    def duplicate_check(self, profile_id: str, account_id: str, image_url: str) -> tuple[float, bool]:
        existing = self.db.get_representation(profile_id)
        if not existing or existing.account_id != account_id or existing.status != "ready" or not existing.face_template:
            raise LookupError("a ready representation owned by this account is required")
        image = self.fetcher.fetch(image_url)
        assess_quality(image, self.fetcher.settings)
        _, template = self.engine.representations(image)
        prior = np.asarray(existing.face_template, dtype=np.float64)
        similarity = float(np.dot(prior, template) / (np.linalg.norm(prior) * np.linalg.norm(template)))
        # This conservative operational threshold must be calibrated per supplied face artifact before broad rollout.
        return similarity, similarity >= 0.88

    def rank(self, viewer_id: str, request_id: str, candidate_ids: list[str], limit: int) -> tuple[list[tuple[str, float | None]], list[str], str]:
        unique_ids = list(dict.fromkeys(candidate_ids))
        representations = self.db.ready_representations(unique_ids)
        unavailable = [candidate_id for candidate_id in unique_ids if candidate_id not in representations]
        eligible = [candidate_id for candidate_id in unique_ids if candidate_id in representations]
        projection = self.artifacts.projection
        stored = self.db.get_user_model(viewer_id)
        if not projection or not stored or stored.projection_version != projection.version or stored.n_observations == 0:
            ordered = neutral_order(viewer_id, request_id, eligible)[:limit]
            return [(candidate_id, None) for candidate_id in ordered], unavailable, "neutral"
        state = BayesianState.from_payload(stored.precision, stored.information, stored.n_observations)
        scored = [(candidate_id, state.score(project(representations[candidate_id].recommendation_embedding or [], projection))) for candidate_id in eligible]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:limit], unavailable, "personalized"

    def swipe(self, event_id: str, viewer_id: str, candidate_id: str, reaction: str, impression_id: str) -> tuple[bool, bool]:
        projection = self.artifacts.projection
        representation = self.db.get_representation(candidate_id)

        def update_model(stored: UserModel | None) -> UserModel | None:
            if not (
                projection
                and representation
                and representation.status == "ready"
                and representation.recommendation_embedding
            ):
                return None
            if stored and stored.projection_version == projection.version:
                state = BayesianState.from_payload(
                    stored.precision, stored.information, stored.n_observations
                )
            else:
                state = BayesianState.empty(len(projection.components))
            state.update(project(representation.recommendation_embedding, projection), reaction)
            precision, information, n_observations = state.payload()
            return UserModel(
                viewer_id=viewer_id,
                projection_version=projection.version,
                precision=precision,
                information=information,
                n_observations=n_observations,
            )

        return self.db.record_swipe(
            event_id,
            viewer_id,
            candidate_id,
            reaction,
            impression_id,
            update_model,
        )
