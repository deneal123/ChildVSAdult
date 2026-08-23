from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    or_,
    select,
    text,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Representation(Base):
    __tablename__ = "prom_profile_representations"

    profile_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    account_id: Mapped[str] = mapped_column(String(128), index=True)
    photo_version: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(24), index=True, default="pending")
    recommendation_embedding: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    face_template: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    vision_model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    face_model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    quality: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class ProcessingJob(Base):
    __tablename__ = "prom_processing_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    profile_id: Mapped[str] = mapped_column(ForeignKey("prom_profile_representations.profile_id"), index=True)
    account_id: Mapped[str] = mapped_column(String(128), index=True)
    photo_version: Mapped[str] = mapped_column(String(128))
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(24), default="pending", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class UserModel(Base):
    __tablename__ = "prom_user_models"

    viewer_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    projection_version: Mapped[str] = mapped_column(String(128))
    precision: Mapped[list[float]] = mapped_column(JSON)
    information: Mapped[list[float]] = mapped_column(JSON)
    n_observations: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class SwipeEvent(Base):
    __tablename__ = "prom_swipe_events"

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    viewer_id: Mapped[str] = mapped_column(String(128), index=True)
    candidate_id: Mapped[str] = mapped_column(String(128))
    reaction: Mapped[str] = mapped_column(String(8))
    impression_id: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuditEvent(Base):
    __tablename__ = "prom_audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    subject_id: Mapped[str] = mapped_column(String(128), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SchemaMigration(Base):
    __tablename__ = "prom_schema_migrations"

    version: Mapped[str] = mapped_column(String(64), primary_key=True)
    applied_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class OwnershipConflict(ValueError):
    pass


@dataclass(frozen=True)
class ClaimedJob:
    id: int
    profile_id: str
    account_id: str
    photo_version: str
    source_url: str


class Database:
    def __init__(self, url: str):
        self.engine = create_engine(url, pool_pre_ping=True)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def apply_migrations(self) -> list[str]:
        """Apply registered, forward-only prom migrations.

        The first migration creates only the isolated ``prom_*`` tables. Further migrations are
        added explicitly to this registry; a deployed database never relies on application startup
        to mutate schema.
        """
        Base.metadata.create_all(self.engine)
        with self.session() as session:
            applied = {row.version for row in session.execute(select(SchemaMigration)).scalars()}
            if "001_initial" not in applied:
                session.add(SchemaMigration(version="001_initial"))
                return ["001_initial"]
        return []

    def create_schema(self) -> None:
        """Compatibility helper used only by tests."""
        self.apply_migrations()

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self.sessions()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def queue_representation(self, profile_id: str, account_id: str, photo_version: str, source_url: str) -> None:
        with self.session() as session:
            representation = session.get(Representation, profile_id)
            if representation and representation.account_id != account_id:
                raise OwnershipConflict("profile account ownership cannot change in prom")
            if representation is None:
                representation = Representation(
                    profile_id=profile_id,
                    account_id=account_id,
                    photo_version=photo_version,
                    status="pending",
                )
                session.add(representation)
            else:
                representation.photo_version = photo_version
                representation.status = "pending"
                representation.error_code = None
            # PostgreSQL enforces the job foreign key immediately; make a new representation visible first.
            session.flush()
            session.add(
                ProcessingJob(
                    profile_id=profile_id,
                    account_id=account_id,
                    photo_version=photo_version,
                    source_url=source_url,
                    status="pending",
                )
            )
            session.add(AuditEvent(event_type="representation_queued", subject_id=profile_id))

    def get_representation(self, profile_id: str) -> Representation | None:
        with self.session() as session:
            return session.get(Representation, profile_id)

    def claim_job(self, lease_seconds: int) -> ClaimedJob | None:
        with self.session() as session:
            now = utcnow()
            statement = (
                select(ProcessingJob)
                .where(
                    or_(
                        ProcessingJob.status == "pending",
                        (ProcessingJob.status == "processing") & (ProcessingJob.lease_until < now),
                    )
                )
                .order_by(ProcessingJob.id)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            job = session.execute(statement).scalar_one_or_none()
            if job is None:
                return None
            if not job.source_url:
                job.status = "failed"
                job.error_code = "missing_source"
                return None
            job.status = "processing"
            job.attempts += 1
            job.lease_until = now + timedelta(seconds=lease_seconds)
            return ClaimedJob(job.id, job.profile_id, job.account_id, job.photo_version, job.source_url)

    def complete_job(
        self,
        job: ClaimedJob,
        recommendation_embedding: list[float],
        face_template: list[float],
        model_version: str,
        quality: dict[str, float | int],
    ) -> None:
        with self.session() as session:
            item = session.get(ProcessingJob, job.id)
            representation = session.get(Representation, job.profile_id)
            if not item or not representation or item.status != "processing":
                return
            # A newer photo can have been queued while this job was running; never overwrite it.
            if representation.photo_version != job.photo_version:
                item.status = "superseded"
                item.source_url = None
                item.lease_until = None
                return
            representation.recommendation_embedding = recommendation_embedding
            representation.face_template = face_template
            representation.vision_model_version = model_version
            representation.face_model_version = model_version
            representation.quality = quality
            representation.status = "ready"
            representation.error_code = None
            item.status = "succeeded"
            item.source_url = None
            item.lease_until = None
            session.add(AuditEvent(event_type="representation_ready", subject_id=job.profile_id))

    def fail_job(self, job: ClaimedJob, error_code: str, retryable: bool, max_attempts: int) -> None:
        with self.session() as session:
            item = session.get(ProcessingJob, job.id)
            representation = session.get(Representation, job.profile_id)
            if not item or item.status != "processing":
                return
            should_retry = retryable and item.attempts < max_attempts and bool(item.source_url)
            item.error_code = error_code
            item.lease_until = None
            item.status = "pending" if should_retry else "failed"
            if not should_retry:
                item.source_url = None
            if representation and representation.photo_version == job.photo_version:
                representation.status = "pending" if should_retry else "failed"
                representation.error_code = None if should_retry else error_code
            event_type = "representation_retry" if should_retry else "representation_failed"
            session.add(AuditEvent(event_type=event_type, subject_id=job.profile_id))

    def ready_representations(self, candidate_ids: list[str]) -> dict[str, Representation]:
        if not candidate_ids:
            return {}
        with self.session() as session:
            rows = session.execute(
                select(Representation).where(
                    Representation.profile_id.in_(candidate_ids), Representation.status == "ready"
                )
            ).scalars()
            return {row.profile_id: row for row in rows if row.recommendation_embedding is not None}

    def get_user_model(self, viewer_id: str) -> UserModel | None:
        with self.session() as session:
            return session.get(UserModel, viewer_id)

    def record_swipe(
        self,
        event_id: str,
        viewer_id: str,
        candidate_id: str,
        reaction: str,
        impression_id: str,
        update_model: Callable[[UserModel | None], UserModel | None],
    ) -> tuple[bool, bool]:
        """Persist event and preference state in one serialised transaction."""
        try:
            with self.session() as session:
                if self.engine.dialect.name == "postgresql":
                    # Serialises first-event creation and every later posterior update per viewer.
                    session.execute(text("SELECT pg_advisory_xact_lock(hashtext(:viewer_id))"), {"viewer_id": viewer_id})
                if session.get(SwipeEvent, event_id):
                    return False, False
                existing = session.execute(
                    select(UserModel).where(UserModel.viewer_id == viewer_id).with_for_update()
                ).scalar_one_or_none()
                model = update_model(existing)
                session.add(
                    SwipeEvent(
                        event_id=event_id,
                        viewer_id=viewer_id,
                        candidate_id=candidate_id,
                        reaction=reaction,
                        impression_id=impression_id,
                    )
                )
                if model is not None:
                    if existing is None:
                        session.add(model)
                    else:
                        existing.projection_version = model.projection_version
                        existing.precision = model.precision
                        existing.information = model.information
                        existing.n_observations = model.n_observations
                session.add(AuditEvent(event_type="swipe_recorded", subject_id=viewer_id))
            return True, model is not None
        except IntegrityError:
            # Concurrent delivery of the same event is harmless and must not update twice.
            return False, False

    def erase_account(self, account_id: str) -> int:
        with self.session() as session:
            representations = session.execute(
                select(Representation).where(Representation.account_id == account_id)
            ).scalars().all()
            profile_ids = [row.profile_id for row in representations]
            if profile_ids:
                for job in session.execute(
                    select(ProcessingJob).where(ProcessingJob.profile_id.in_(profile_ids))
                ).scalars():
                    session.delete(job)
                session.flush()
            for row in representations:
                session.delete(row)
            session.get(UserModel, account_id) and session.delete(session.get(UserModel, account_id))
            for event in session.execute(select(SwipeEvent).where(SwipeEvent.viewer_id == account_id)).scalars():
                session.delete(event)
            session.add(AuditEvent(event_type="account_erased", subject_id=account_id))
            return len(profile_ids)
