.PHONY: install data ingest run test lint eval docker

install:
	pip install -e ".[dev]"

data:
	python scripts/generate_sample_data.py

ingest:
	python -m app.ingestion.pipeline data/sample_docs

run: ingest
	uvicorn app.api.main:app --reload

test:
	pytest -q

lint:
	ruff check src tests scripts eval

eval:
	python eval/run_eval.py --generation

docker:
	docker compose up --build
