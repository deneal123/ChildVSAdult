from __future__ import annotations


def test_migration_is_versioned_and_idempotent(service):
    assert service.db.apply_migrations() == []


def test_retryable_job_keeps_source_only_until_terminal_failure(service):
    service.db.queue_representation("p1", "a1", "v1", "https://media.internal.example/photo.jpg")
    first = service.db.claim_job(lease_seconds=1)
    assert first is not None
    service.db.fail_job(first, "inference_failed", retryable=True, max_attempts=3)
    assert service.db.get_representation("p1").status == "pending"

    second = service.db.claim_job(lease_seconds=1)
    assert second is not None
    service.db.fail_job(second, "media_rejected", retryable=False, max_attempts=3)
    representation = service.db.get_representation("p1")
    assert representation is not None and representation.status == "failed"
