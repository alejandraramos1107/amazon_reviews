FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Install locked dependencies first for better layer caching.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application source.
COPY app.py deploy.py pipeline.py README.md ./
COPY src ./src
COPY templates ./templates
COPY data ./data
COPY serving_artifacts ./serving_artifacts

# Make the virtualenv available to the runtime process.
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
