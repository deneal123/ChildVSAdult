from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

import numpy as np

from .artifacts import Projection


class RecommendationError(ValueError):
    pass


@dataclass
class BayesianState:
    precision: np.ndarray
    information: np.ndarray
    n_observations: int

    @classmethod
    def empty(cls, dimension: int, prior_precision: float = 30.0) -> BayesianState:
        return cls(np.eye(dimension, dtype=np.float64) * prior_precision, np.zeros(dimension), 0)

    @classmethod
    def from_payload(cls, precision: list[float], information: list[float], n_observations: int) -> BayesianState:
        d = len(information)
        matrix = np.asarray(precision, dtype=np.float64)
        if d == 0 or matrix.size != d * d:
            raise RecommendationError("stored preference state has an invalid dimension")
        return cls(matrix.reshape(d, d), np.asarray(information, dtype=np.float64), n_observations)

    @property
    def mean(self) -> np.ndarray:
        return np.linalg.solve(self.precision, self.information)

    def update(self, feature: np.ndarray, reaction: str) -> None:
        if feature.shape != self.information.shape:
            raise RecommendationError("feature dimension does not match preference state")
        target = 1.0 if reaction == "like" else -1.0
        self.precision += np.outer(feature, feature)
        self.information += feature * target
        self.n_observations += 1

    def score(self, feature: np.ndarray) -> float:
        covariance_feature = np.linalg.solve(self.precision, feature)
        uncertainty = float(np.sqrt(max(0.0, feature @ covariance_feature)))
        # A small deterministic uncertainty bonus prevents premature feed collapse.
        return float(feature @ self.mean + 0.15 * uncertainty)

    def payload(self) -> tuple[list[float], list[float], int]:
        return self.precision.reshape(-1).tolist(), self.information.tolist(), self.n_observations


def project(vector: list[float], projection: Projection) -> np.ndarray:
    return projection.transform(np.asarray(vector, dtype=np.float64))


def neutral_order(viewer_id: str, request_id: str, candidate_ids: list[str]) -> list[str]:
    """Stable pseudo-random order; no attractiveness or hidden profile signal is used."""
    seed = f"{viewer_id}:{request_id}".encode()

    def key(candidate_id: str) -> bytes:
        return hmac.new(seed, candidate_id.encode("utf-8"), hashlib.sha256).digest()

    return sorted(candidate_ids, key=key)
