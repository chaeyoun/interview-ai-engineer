FROM python:3.13.2-slim

COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /bin/uv

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1

COPY fda-regulations/pyproject.toml fda-regulations/uv.lock fda-regulations/README.md ./

RUN uv sync --frozen --no-install-project --no-dev

COPY fda-regulations/ ./

RUN uv sync --frozen --no-dev && chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["/app/.venv/bin/python", "main.py"]
CMD ["--target-year", "2026"]