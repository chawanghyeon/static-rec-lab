UV ?= uv
PYTHON_TARGETS := apps recsys scripts tests
RAW_RATINGS ?= data/raw/ml-latest-small/ratings.csv
MOVIELENS_RAW_DIR ?= data/raw
MOVIELENS_DATASET ?= ml-latest-small
MOVIELENS_URL ?= https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
PROCESSED_DIR ?= data/processed
BASELINE_DIR ?= artifacts/baseline
BASELINE_REPORT ?= reports/baseline.md
ITEM_KNN_MAX_CANDIDATES ?= 200
ITEM_KNN_MAX_HISTORY_ITEMS ?= 50
MIN_INTERACTIONS ?= 5
MAX_HISTORY_LENGTH ?= 50

.PHONY: format lint test download-movielens preprocess train-baseline eval-baseline benchmark-decoder check

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

train-baseline:
	$(UV) run python scripts/train_baseline.py \
		--train-parquet $(PROCESSED_DIR)/train.parquet \
		--output-dir $(BASELINE_DIR) \
		--max-candidates-per-item $(ITEM_KNN_MAX_CANDIDATES) \
		--max-history-items $(ITEM_KNN_MAX_HISTORY_ITEMS)

eval-baseline:
	$(UV) run python scripts/eval_baseline.py \
		--model-dir $(BASELINE_DIR) \
		--valid-parquet $(PROCESSED_DIR)/valid.parquet \
		--test-parquet $(PROCESSED_DIR)/test.parquet \
		--report-path $(BASELINE_REPORT)

benchmark-decoder:
	$(UV) run python scripts/benchmark_decoder.py

check: lint test
