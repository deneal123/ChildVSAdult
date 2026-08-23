FROM python:3.12-slim AS model-init

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
# The init container only exports ONNX; use the CPU wheel and keep CUDA out of the image.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu ".[model-init]"
COPY docker/model-init-entrypoint.sh /usr/local/bin/model-init-entrypoint
RUN chmod 755 /usr/local/bin/model-init-entrypoint
RUN groupadd --gid 10001 prom && useradd --uid 10001 --gid 10001 --create-home prom
WORKDIR /app
ENTRYPOINT ["/usr/local/bin/model-init-entrypoint"]

FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --wheel-dir /wheels .

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN groupadd --gid 10001 prom && useradd --uid 10001 --gid 10001 --create-home prom
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels
USER prom
EXPOSE 8080
CMD ["prom-api"]
