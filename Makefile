UV ?= uv
PYTHON_TARGETS := apps recsys scripts tests
RAW_RATINGS ?= data/raw/ratings.csv
PROCESSED_DIR ?= data/processed
MIN_INTERACTIONS ?= 5
MAX_HISTORY_LENGTH ?= 50

.PHONY: format lint test preprocess benchmark-decoder check

format:
	$(UV) run ruff format $(PYTHON_TARGETS)
	$(UV) run ruff check --fix $(PYTHON_TARGETS)

lint:
	$(UV) run ruff format --check $(PYTHON_TARGETS)
	$(UV) run ruff check $(PYTHON_TARGETS)
	$(UV) run mypy

test:
	$(UV) run pytest

preprocess:
	$(UV) run python scripts/preprocess.py \
		--ratings-csv $(RAW_RATINGS) \
		--output-dir $(PROCESSED_DIR) \
		--min-interactions $(MIN_INTERACTIONS) \
		--max-history-length $(MAX_HISTORY_LENGTH)

benchmark-decoder:
	$(UV) run python scripts/benchmark_decoder.py

check: lint test
