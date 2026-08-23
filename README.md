# prom — internal dating recommendation service

`prom` is an internal CPU-first service. The dating backend remains responsible for authentication,
18+ eligibility, mutual preferences, blocks, distance, rate limits and media approval. It supplies
only eligible opaque profile IDs to `prom`; browser clients must never call this service directly.

The service stores only derived embeddings, same-account face templates, feedback state and a minimal
audit trail. It never stores source images or searches a face across other users.

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

`prom-model-init` materialises two separate approved models once in the persistent Docker volume:
CLIP ViT-B/32 is the frozen visual embedding model used by ranking, while AdaFace IR-50 produces
same-profile duplicate templates. It verifies every downloaded source file by SHA-256, exports CPU
ONNX, checks PyTorch/ONNX parity and writes the manifest. API and worker mount that volume read-only.
Supply a prebuilt manifest instead only when it passes `prom-verify-artifacts`.

```powershell
uv run prom-model-init
uv run prom-verify-artifacts
uv run prom-migrate
uv run prom-api
uv run prom-worker
```

The bootstrap creates a deterministic orthogonal projection without fitting on research or user data;
it enables Bayesian personalization from the first swipe. Replace it only with a separately reviewed,
versioned whitening artifact trained on a consented production corpus. The service returns a
deterministic neutral ordering until reactions exist.

## Deployment

Copy `.env.example` to the deployment secret store and run
`docker compose up prom-model-init prom-migrate prom-api prom-worker`. The one-shot model-init service
must finish successfully before migration, API or worker starts. `docker-compose.yml` intentionally does not
create a PostgreSQL instance: it connects to the application's managed database using its own
`prom_*` tables.
