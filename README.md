# prom — internal dating recommendation service

`prom` is an internal CPU-first service. The dating backend remains responsible for authentication,
18+ eligibility, mutual preferences, blocks, distance, rate limits and media approval. It supplies
only eligible opaque profile IDs to `prom`; browser clients must never call this service directly.

The service stores only derived embeddings, same-account face templates, feedback state and a minimal
audit trail. It never stores source images, downloads models at runtime, or searches a face across
other users.

## Contract

- `PUT /v1/profiles/{profile_id}/representation` queues an approved-photo URL for processing.
- `GET /v1/profiles/{profile_id}/representation` returns processing state.
- `POST /v1/rank` ranks only the supplied eligible candidate IDs.
- `POST /v1/swipes` records an idempotent `like` or `pass` and updates the viewer model.
- `POST /v1/profiles/{profile_id}/duplicate-check` compares a new image only to that profile's
  existing template, with explicit consent.
- `DELETE /v1/accounts/{account_id}` erases all derived data belonging to that account.

All `/v1/*` endpoints require `Authorization: Bearer $PROM_INTERNAL_TOKEN`.

## Models

Mount a read-only manifest and its two ONNX artifacts at `PROM_MODEL_MANIFEST`. Generate checksums
after retrieving approved, licensed artifacts from the organisation's model registry:

```powershell
uv run prom-verify-artifacts
uv run prom-migrate
uv run prom-api
uv run prom-worker
```

The optional projection artifact is required before personalised ranking is enabled. Until then the
service returns a deterministic, neutral ordering of the backend-supplied pool.

## Deployment

Copy `.env.example` to the deployment secret store, provide the model manifest and run
`docker compose up prom-migrate prom-api prom-worker`. `docker-compose.yml` intentionally does not
create a PostgreSQL instance: it connects to the application's managed database using its own
`prom_*` tables.
