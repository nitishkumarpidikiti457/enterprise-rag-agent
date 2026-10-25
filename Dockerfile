# ---- build stage: install dependencies and download models ----
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 FASTEMBED_CACHE_PATH=/models
RUN python -m venv /venv
ENV PATH=/venv/bin:$PATH
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Pre-download the free local embedding + re-ranker models so the container starts fast.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5'); \
from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"

# ---- runtime stage: only what is needed to run ----
FROM python:3.12-slim
RUN useradd --create-home app
COPY --from=build /venv /venv
COPY --from=build /models /models
ENV PATH=/venv/bin:$PATH PYTHONUNBUFFERED=1 FASTEMBED_CACHE_PATH=/models
WORKDIR /app
COPY --chown=app:app data ./data
COPY --chown=app:app eval ./eval
RUN mkdir -p .chroma .state && chown -R app:app /app /models
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["sh", "-c", "python -m app.ingestion.pipeline data/sample_docs && uvicorn app.api.main:app --host 0.0.0.0 --port 8000"]
