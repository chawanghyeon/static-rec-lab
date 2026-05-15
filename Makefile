UV ?= uv
PYTHON_TARGETS := apps recsys scripts tests
RAW_RATINGS ?= data/raw/ml-latest-small/ratings.csv
MOVIELENS_RAW_DIR ?= data/raw
MOVIELENS_DATASET ?= ml-latest-small
MOVIELENS_URL ?= https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
API_HOST ?= 0.0.0.0
API_PORT ?= 8000
PROCESSED_DIR ?= data/processed
BASELINE_DIR ?= artifacts/baseline
BASELINE_REPORT ?= reports/baseline.md
ITEM_KNN_MAX_CANDIDATES ?= 200
ITEM_KNN_MAX_HISTORY_ITEMS ?= 50
SEMANTIC_ID_PATH ?= artifacts/semantic_id/semantic_ids.json
SEMANTIC_ID_REPORT ?= reports/semantic_id.md
SEMANTIC_ID_DEPTH ?= 4
SEMANTIC_ID_BRANCHING_FACTOR ?= 16
SEMANTIC_ID_COMPONENTS ?= 32
GENERATIVE_CHECKPOINT ?= artifacts/generative/model.pt
GENERATIVE_REPORT ?= reports/generative.md
GENERATIVE_RANKING_REPORT ?= reports/generative_eval.md
GENERATIVE_EPOCHS ?= 1
GENERATIVE_BATCH_SIZE ?= 64
GENERATIVE_LR ?= 0.001
GENERATIVE_BEAM_SIZE ?= 50
DECODER_BENCHMARK_REPORT ?= reports/decoder_benchmark.md
DECODER_BENCHMARK_BATCH_SIZES ?= 1 32 128 512
SERVING_BENCHMARK_REPORT ?= reports/serving_benchmark.md
SERVING_BENCHMARK_BATCH_SIZES ?= 1 32 128
SERVING_BENCHMARK_SERVICE ?= mock
MIN_INTERACTIONS ?= 5
MAX_HISTORY_LENGTH ?= 50

.PHONY: format lint test download-movielens preprocess train-baseline eval-baseline build-semantic-ids validate-semantic-ids train-generative eval-generative eval-generative-ranking benchmark-decoder benchmark-serving serve-api check

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

build-semantic-ids:
	$(UV) run python scripts/build_semantic_ids.py \
		--train-parquet $(PROCESSED_DIR)/train.parquet \
		--output-path $(SEMANTIC_ID_PATH) \
		--report-path $(SEMANTIC_ID_REPORT) \
		--depth $(SEMANTIC_ID_DEPTH) \
		--branching-factor $(SEMANTIC_ID_BRANCHING_FACTOR) \
		--n-components $(SEMANTIC_ID_COMPONENTS) \
		--max-history-items $(MAX_HISTORY_LENGTH)

validate-semantic-ids:
	$(UV) run python scripts/validate_semantic_ids.py \
		--semantic-id-path $(SEMANTIC_ID_PATH)

train-generative:
	$(UV) run python scripts/train_generative.py \
		--train-parquet $(PROCESSED_DIR)/train.parquet \
		--valid-parquet $(PROCESSED_DIR)/valid.parquet \
		--semantic-id-path $(SEMANTIC_ID_PATH) \
		--output-path $(GENERATIVE_CHECKPOINT) \
		--epochs $(GENERATIVE_EPOCHS) \
		--batch-size $(GENERATIVE_BATCH_SIZE) \
		--learning-rate $(GENERATIVE_LR) \
		--max-history-length $(MAX_HISTORY_LENGTH)

eval-generative:
	$(UV) run python scripts/eval_generative.py \
		--checkpoint-path $(GENERATIVE_CHECKPOINT) \
		--eval-parquet $(PROCESSED_DIR)/valid.parquet \
		--semantic-id-path $(SEMANTIC_ID_PATH) \
		--report-path $(GENERATIVE_REPORT) \
		--batch-size $(GENERATIVE_BATCH_SIZE)

eval-generative-ranking:
	$(UV) run python scripts/eval_generative_ranking.py \
		--checkpoint-path $(GENERATIVE_CHECKPOINT) \
		--semantic-id-path $(SEMANTIC_ID_PATH) \
		--valid-parquet $(PROCESSED_DIR)/valid.parquet \
		--test-parquet $(PROCESSED_DIR)/test.parquet \
		--report-path $(GENERATIVE_RANKING_REPORT) \
		--beam-size $(GENERATIVE_BEAM_SIZE)

benchmark-decoder:
	$(UV) run python scripts/benchmark_decoder.py \
		--report-path $(DECODER_BENCHMARK_REPORT) \
		--batch-sizes $(DECODER_BENCHMARK_BATCH_SIZES)

benchmark-serving:
	$(UV) run python scripts/benchmark_serving.py \
		--service $(SERVING_BENCHMARK_SERVICE) \
		--report-path $(SERVING_BENCHMARK_REPORT) \
		--batch-sizes $(SERVING_BENCHMARK_BATCH_SIZES)

serve-api:
	$(UV) run uvicorn apps.api.main:app --host $(API_HOST) --port $(API_PORT) --reload

check: lint test
