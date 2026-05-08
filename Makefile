UV ?= uv
PYTHON_TARGETS := apps recsys scripts tests
RAW_RATINGS ?= data/raw/ratings.csv
MOVIELENS_RAW_DIR ?= data/raw
MOVIELENS_DATASET ?= ml-latest-small
MOVIELENS_URL ?= https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
PROCESSED_DIR ?= data/processed
MIN_INTERACTIONS ?= 5
MAX_HISTORY_LENGTH ?= 50

.PHONY: format lint test download-movielens preprocess benchmark-decoder check

format:
	$(UV) run ruff format $(PYTHON_TARGETS)
	$(UV) run ruff check --fix $(PYTHON_TARGETS)

lint:
	$(UV) run ruff format --check $(PYTHON_TARGETS)
	$(UV) run ruff check $(PYTHON_TARGETS)
	$(UV) run mypy

test:
	$(UV) run pytest

download-movielens:
	$(UV) run python scripts/download_movielens.py \
		--output-dir $(MOVIELENS_RAW_DIR) \
		--dataset-name $(MOVIELENS_DATASET) \
		--url $(MOVIELENS_URL)

preprocess:
	$(UV) run python scripts/preprocess.py \
		--ratings-csv $(RAW_RATINGS) \
		--output-dir $(PROCESSED_DIR) \
		--min-interactions $(MIN_INTERACTIONS) \
		--max-history-length $(MAX_HISTORY_LENGTH)

benchmark-decoder:
	$(UV) run python scripts/benchmark_decoder.py

check: lint test
