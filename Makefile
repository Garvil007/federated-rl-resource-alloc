.PHONY: install test lint train-local train-federated serve monitor stack clean help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/
	mypy src/ --ignore-missing-imports

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-fast:
	pytest tests/ -v --tb=short -q

train-local:
	python scripts/train_local.py

train-federated:
	python scripts/train_federated.py

serve:
	uvicorn src.mlops.serving:app --host 0.0.0.0 --port 8080 --reload

monitor:
	cd monitoring && docker-compose -f docker-compose.monitoring.yml up -d

stack:
	cd docker && docker-compose up -d

clean:
	rm -rf ray_results/ outputs/ wandb/ .mypy_cache/ __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
