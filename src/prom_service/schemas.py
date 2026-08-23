from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl

OpaqueId = str


class RepresentationUpsert(BaseModel):
    account_id: OpaqueId = Field(min_length=1, max_length=128)
    photo_version: str = Field(min_length=1, max_length=128)
    image_url: HttpUrl


class RepresentationStatus(BaseModel):
    profile_id: OpaqueId
    photo_version: str
    status: Literal["pending", "processing", "ready", "failed"]
    error_code: str | None = None


class RankRequest(BaseModel):
    viewer_id: OpaqueId = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)
    candidate_ids: list[OpaqueId] = Field(min_length=1, max_length=500)
    limit: int = Field(default=50, ge=1, le=100)


class RankedCandidate(BaseModel):
    profile_id: OpaqueId
    score: float | None = None


class RankResponse(BaseModel):
    candidates: list[RankedCandidate]
    unavailable_candidate_ids: list[OpaqueId]
    mode: Literal["neutral", "personalized"]


class SwipeRequest(BaseModel):
    event_id: str = Field(min_length=1, max_length=128)
    viewer_id: OpaqueId = Field(min_length=1, max_length=128)
    candidate_id: OpaqueId = Field(min_length=1, max_length=128)
    reaction: Literal["like", "pass"]
    impression_id: str = Field(min_length=1, max_length=128)


class SwipeResponse(BaseModel):
    applied: bool
    personalization_updated: bool


class DuplicateCheckRequest(BaseModel):
    account_id: OpaqueId = Field(min_length=1, max_length=128)
    image_url: HttpUrl
    biometric_consent: Literal[True]


class DuplicateCheckResponse(BaseModel):
    profile_id: OpaqueId
    comparison_scope: Literal["same_profile_only"]
    similarity: float
    is_likely_duplicate: bool


class EraseResponse(BaseModel):
    erased_profile_representations: int
