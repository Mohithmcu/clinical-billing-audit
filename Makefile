.PHONY: install install-dev install-all clean test lint format run-pipeline export-powerbi help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install base dependencies
	pip install -e .

install-dev:  ## Install with dev tools (pytest, ruff, mypy)
	pip install -e ".[dev]"

install-all:  ## Install everything including transformers
	pip install -e ".[all]"

install-spacy-models:  ## Download spaCy/scispaCy models
	python -m spacy download en_core_web_sm
	pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
	pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

clean:  ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .coverage htmlcov/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

test:  ## Run tests with coverage
	pytest tests/ -v --cov=clinical_billing_audit --cov-report=term-missing

test-quick:  ## Run tests without coverage
	pytest tests/ -v --tb=short

lint:  ## Run linter
	ruff check src/ tests/

format:  ## Format code
	ruff format src/ tests/

typecheck:  ## Run type checker
	mypy src/clinical_billing_audit/

run-pipeline:  ## Run the full pipeline
	python scripts/run_pipeline.py --config configs/default.yaml

export-powerbi:  ## Export data for Power BI dashboard
	python scripts/export_powerbi_data.py

data-quality:  ## Run data quality checks only
	cba data-quality --config configs/default.yaml

extract-entities:  ## Run NER extraction only
	cba extract-entities --config configs/default.yaml

train-baseline:  ## Train baseline model only
	cba train --model baseline --config configs/default.yaml

detect-anomalies:  ## Run anomaly detection
	cba detect-anomalies --config configs/default.yaml

evaluate:  ## Run evaluation
	cba evaluate --config configs/default.yaml
