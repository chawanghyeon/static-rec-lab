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
BASELINE_REPORT ?= reports/local/baseline.md
ITEM_KNN_MAX_CANDIDATES ?= 200
ITEM_KNN_MAX_HISTORY_ITEMS ?= 50
SEMANTIC_ID_PATH ?= artifacts/semantic_id/semantic_ids.json
SEMANTIC_ID_REPORT ?= reports/local/semantic_id.md
STATIC_DECODING_INDEX_PATH ?= artifacts/semantic_id/static_decoding_index.npz
STATIC_DECODING_DENSE_LOOKUP_LAYERS ?= auto
ifneq ($(wildcard $(STATIC_DECODING_INDEX_PATH)),)
STATIC_DECODING_INDEX_CLI_ARG := --static-decoding-index-path $(STATIC_DECODING_INDEX_PATH)
else
STATIC_DECODING_INDEX_CLI_ARG :=
endif
SEMANTIC_ID_DEPTH ?= 4
SEMANTIC_ID_BRANCHING_FACTOR ?= 16
SEMANTIC_ID_COMPONENTS ?= 32
GENERATIVE_CHECKPOINT ?= artifacts/generative/model.pt
GENERATIVE_REPORT ?= reports/local/generative.md
GENERATIVE_RANKING_REPORT ?= reports/local/generative_eval.md
GENERATIVE_EPOCHS ?= 1
GENERATIVE_EVAL_EVERY_EPOCHS ?= 1
GENERATIVE_SKIP_VALID_ACCURACY ?= 0
GENERATIVE_BATCH_SIZE ?= 64
GENERATIVE_LR ?= 0.001
GENERATIVE_BEAM_SIZE ?= 50
GENERATIVE_INFERENCE_BATCH_SIZE ?= 128
GENERATIVE_DEVICE ?= auto
GENERATIVE_PARQUET_BATCH_SIZE ?= 65536
GENERATIVE_NUM_WORKERS ?= 0
GENERATIVE_PREFETCH_FACTOR ?= 2
GENERATIVE_LOG_EVERY_BATCHES ?= 250
GENERATIVE_COMPILE_MODEL ?= 0
GENERATIVE_AMP ?= 0
GENERATIVE_AMP_DTYPE ?= float16
ifneq ($(filter 1 true yes,$(GENERATIVE_COMPILE_MODEL)),)
GENERATIVE_COMPILE_MODEL_CLI_ARG := --compile-model
else
GENERATIVE_COMPILE_MODEL_CLI_ARG :=
endif
ifneq ($(filter 1 true yes,$(GENERATIVE_AMP)),)
GENERATIVE_AMP_CLI_ARG := --amp --amp-dtype $(GENERATIVE_AMP_DTYPE)
else
GENERATIVE_AMP_CLI_ARG :=
endif
ifneq ($(filter 1 true yes,$(GENERATIVE_SKIP_VALID_ACCURACY)),)
GENERATIVE_SKIP_VALID_ACCURACY_CLI_ARG := --skip-valid-accuracy
else
GENERATIVE_SKIP_VALID_ACCURACY_CLI_ARG :=
endif
DECODER_BENCHMARK_REPORT ?= reports/local/decoder_benchmark.md
DECODER_BENCHMARK_BATCH_SIZES ?= 1 32 128 512
DECODER_BENCHMARK_SEMANTIC_ID_PATH ?=
ifneq ($(strip $(DECODER_BENCHMARK_SEMANTIC_ID_PATH)),)
DECODER_BENCHMARK_SEMANTIC_ID_CLI_ARG := --semantic-id-path $(DECODER_BENCHMARK_SEMANTIC_ID_PATH)
else
DECODER_BENCHMARK_SEMANTIC_ID_CLI_ARG :=
endif
SERVING_BENCHMARK_REPORT ?= reports/local/serving_benchmark.md
HTTP_SERVING_BENCHMARK_REPORT ?= reports/local/http_serving_benchmark.md
SERVING_BENCHMARK_BATCH_SIZES ?= 1 32 128
SERVING_BENCHMARK_SERVICE ?= mock
MIN_INTERACTIONS ?= 5
MAX_HISTORY_LENGTH ?= 50

ML32M_URL ?= https://files.grouplens.org/datasets/movielens/ml-32m.zip
ML32M_RAW_RATINGS ?= data/raw/ml-32m/ratings.csv
ML32M_PROCESSED_DIR ?= data/processed/ml-32m
ML32M_BASELINE_DIR ?= artifacts/baseline/ml-32m
ML32M_BASELINE_REPORT ?= reports/ml_32m_baseline.md
ML32M_SEMANTIC_ID_PATH ?= artifacts/semantic_id/ml-32m/semantic_ids.json
ML32M_SEMANTIC_ID_REPORT ?= reports/ml_32m_semantic_id.md
ML32M_STATIC_DECODING_INDEX_PATH ?= artifacts/semantic_id/ml-32m/static_decoding_index.npz
ML32M_GENERATIVE_CHECKPOINT ?= artifacts/generative/ml-32m/model.pt
ML32M_GENERATIVE_REPORT ?= reports/ml_32m_generative.md
ML32M_GENERATIVE_RANKING_REPORT ?= reports/ml_32m_generative_eval.md
ML32M_DECODER_BENCHMARK_REPORT ?= reports/ml_32m_decoder_benchmark.md
ML32M_SERVING_BENCHMARK_REPORT ?= reports/ml_32m_serving_benchmark.md
ML32M_HTTP_SERVING_BENCHMARK_REPORT ?= reports/ml_32m_http_serving_benchmark.md
ML32M_SEMANTIC_ID_BRANCHING_FACTOR ?= 32
ML32M_GENERATIVE_BATCH_SIZE ?= 4096
ML32M_GENERATIVE_BEAM_SIZE ?= 20
ML32M_GENERATIVE_PARQUET_BATCH_SIZE ?= 131072
ML32M_GENERATIVE_NUM_WORKERS ?= 0
ML32M_GENERATIVE_PREFETCH_FACTOR ?= 4
ML32M_GENERATIVE_AMP ?= 0
ML32M_GENERATIVE_AMP_DTYPE ?= float16
ML32M_DEVICE ?= auto
ML32M_SERVING_DEVICE ?= cpu

.PHONY: format lint test download-movielens preprocess train-baseline eval-baseline build-semantic-ids validate-semantic-ids build-static-decoding-index train-generative eval-generative eval-generative-ranking benchmark-decoder benchmark-serving benchmark-api serve-api check download-ml32m preprocess-ml32m baseline-ml32m semantic-ids-ml32m generative-ml32m benchmark-ml32m reproduce-ml32m

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
	$(UV) run python scripts/baseline.py train \
		--train-parquet $(PROCESSED_DIR)/train.parquet \
		--output-dir $(BASELINE_DIR) \
		--max-candidates-per-item $(ITEM_KNN_MAX_CANDIDATES) \
		--max-history-items $(ITEM_KNN_MAX_HISTORY_ITEMS)

eval-baseline:
	$(UV) run python scripts/baseline.py eval \
		--model-dir $(BASELINE_DIR) \
		--valid-parquet $(PROCESSED_DIR)/valid.parquet \
		--test-parquet $(PROCESSED_DIR)/test.parquet \
		--report-path $(BASELINE_REPORT)

build-semantic-ids:
	$(UV) run python scripts/semantic_id.py build \
		--train-parquet $(PROCESSED_DIR)/train.parquet \
		--output-path $(SEMANTIC_ID_PATH) \
		--report-path $(SEMANTIC_ID_REPORT) \
		--depth $(SEMANTIC_ID_DEPTH) \
		--branching-factor $(SEMANTIC_ID_BRANCHING_FACTOR) \
		--n-components $(SEMANTIC_ID_COMPONENTS) \
		--max-history-items $(MAX_HISTORY_LENGTH)

validate-semantic-ids:
	$(UV) run python scripts/semantic_id.py validate \
		--semantic-id-path $(SEMANTIC_ID_PATH)

build-static-decoding-index:
	$(UV) run python scripts/semantic_id.py build-static-index \
		--semantic-id-path $(SEMANTIC_ID_PATH) \
		--output-path $(STATIC_DECODING_INDEX_PATH) \
		--dense-lookup-layers $(STATIC_DECODING_DENSE_LOOKUP_LAYERS)

train-generative:
	$(UV) run python scripts/generative.py train \
		--train-parquet $(PROCESSED_DIR)/train.parquet \
		--valid-parquet $(PROCESSED_DIR)/valid.parquet \
		--semantic-id-path $(SEMANTIC_ID_PATH) \
		--output-path $(GENERATIVE_CHECKPOINT) \
		--epochs $(GENERATIVE_EPOCHS) \
		--eval-every-epochs $(GENERATIVE_EVAL_EVERY_EPOCHS) \
		--batch-size $(GENERATIVE_BATCH_SIZE) \
		--learning-rate $(GENERATIVE_LR) \
		--max-history-length $(MAX_HISTORY_LENGTH) \
		--device $(GENERATIVE_DEVICE) \
		--parquet-batch-size $(GENERATIVE_PARQUET_BATCH_SIZE) \
		--num-workers $(GENERATIVE_NUM_WORKERS) \
		--prefetch-factor $(GENERATIVE_PREFETCH_FACTOR) \
		--log-every-batches $(GENERATIVE_LOG_EVERY_BATCHES) \
		$(GENERATIVE_COMPILE_MODEL_CLI_ARG) \
		$(GENERATIVE_AMP_CLI_ARG) \
		$(GENERATIVE_SKIP_VALID_ACCURACY_CLI_ARG)

eval-generative:
	$(UV) run python scripts/generative.py eval \
		--checkpoint-path $(GENERATIVE_CHECKPOINT) \
		--eval-parquet $(PROCESSED_DIR)/valid.parquet \
		--semantic-id-path $(SEMANTIC_ID_PATH) \
		--report-path $(GENERATIVE_REPORT) \
		--batch-size $(GENERATIVE_BATCH_SIZE) \
		--parquet-batch-size $(GENERATIVE_PARQUET_BATCH_SIZE) \
		--num-workers $(GENERATIVE_NUM_WORKERS) \
		--prefetch-factor $(GENERATIVE_PREFETCH_FACTOR) \
		--device $(GENERATIVE_DEVICE) \
		$(GENERATIVE_AMP_CLI_ARG)

eval-generative-ranking:
	$(UV) run python scripts/generative.py eval-ranking \
		--checkpoint-path $(GENERATIVE_CHECKPOINT) \
		--semantic-id-path $(SEMANTIC_ID_PATH) \
		--valid-parquet $(PROCESSED_DIR)/valid.parquet \
		--test-parquet $(PROCESSED_DIR)/test.parquet \
		--report-path $(GENERATIVE_RANKING_REPORT) \
		--beam-size $(GENERATIVE_BEAM_SIZE) \
		--inference-batch-size $(GENERATIVE_INFERENCE_BATCH_SIZE) \
		--parquet-batch-size $(GENERATIVE_PARQUET_BATCH_SIZE) \
		--device $(GENERATIVE_DEVICE) \
		$(STATIC_DECODING_INDEX_CLI_ARG)

benchmark-decoder:
	$(UV) run python scripts/benchmark.py decoder \
		--report-path $(DECODER_BENCHMARK_REPORT) \
		--batch-sizes $(DECODER_BENCHMARK_BATCH_SIZES) $(DECODER_BENCHMARK_SEMANTIC_ID_CLI_ARG)

benchmark-serving:
	$(UV) run python scripts/benchmark.py serving \
		--service $(SERVING_BENCHMARK_SERVICE) \
		--report-path $(SERVING_BENCHMARK_REPORT) \
		--batch-sizes $(SERVING_BENCHMARK_BATCH_SIZES)

benchmark-api:
	$(UV) run python scripts/benchmark.py serving-http \
		--service $(SERVING_BENCHMARK_SERVICE) \
		--report-path $(HTTP_SERVING_BENCHMARK_REPORT) \
		--batch-sizes $(SERVING_BENCHMARK_BATCH_SIZES)

download-ml32m:
	$(MAKE) download-movielens \
		MOVIELENS_DATASET=ml-32m \
		MOVIELENS_URL=$(ML32M_URL)

preprocess-ml32m:
	$(MAKE) preprocess \
		RAW_RATINGS=$(ML32M_RAW_RATINGS) \
		PROCESSED_DIR=$(ML32M_PROCESSED_DIR)

baseline-ml32m:
	$(MAKE) train-baseline \
		PROCESSED_DIR=$(ML32M_PROCESSED_DIR) \
		BASELINE_DIR=$(ML32M_BASELINE_DIR)
	$(MAKE) eval-baseline \
		PROCESSED_DIR=$(ML32M_PROCESSED_DIR) \
		BASELINE_DIR=$(ML32M_BASELINE_DIR) \
		BASELINE_REPORT=$(ML32M_BASELINE_REPORT)

semantic-ids-ml32m:
	$(MAKE) build-semantic-ids \
		PROCESSED_DIR=$(ML32M_PROCESSED_DIR) \
		SEMANTIC_ID_PATH=$(ML32M_SEMANTIC_ID_PATH) \
		SEMANTIC_ID_REPORT=$(ML32M_SEMANTIC_ID_REPORT) \
		SEMANTIC_ID_BRANCHING_FACTOR=$(ML32M_SEMANTIC_ID_BRANCHING_FACTOR)
	$(MAKE) validate-semantic-ids \
		SEMANTIC_ID_PATH=$(ML32M_SEMANTIC_ID_PATH)
	$(MAKE) build-static-decoding-index \
		SEMANTIC_ID_PATH=$(ML32M_SEMANTIC_ID_PATH) \
		STATIC_DECODING_INDEX_PATH=$(ML32M_STATIC_DECODING_INDEX_PATH)

generative-ml32m:
	$(MAKE) train-generative \
		PROCESSED_DIR=$(ML32M_PROCESSED_DIR) \
		SEMANTIC_ID_PATH=$(ML32M_SEMANTIC_ID_PATH) \
		GENERATIVE_CHECKPOINT=$(ML32M_GENERATIVE_CHECKPOINT) \
		GENERATIVE_BATCH_SIZE=$(ML32M_GENERATIVE_BATCH_SIZE) \
		GENERATIVE_PARQUET_BATCH_SIZE=$(ML32M_GENERATIVE_PARQUET_BATCH_SIZE) \
		GENERATIVE_NUM_WORKERS=$(ML32M_GENERATIVE_NUM_WORKERS) \
		GENERATIVE_PREFETCH_FACTOR=$(ML32M_GENERATIVE_PREFETCH_FACTOR) \
		GENERATIVE_AMP=$(ML32M_GENERATIVE_AMP) \
		GENERATIVE_AMP_DTYPE=$(ML32M_GENERATIVE_AMP_DTYPE) \
		GENERATIVE_DEVICE=$(ML32M_DEVICE)
	$(MAKE) eval-generative \
		PROCESSED_DIR=$(ML32M_PROCESSED_DIR) \
		SEMANTIC_ID_PATH=$(ML32M_SEMANTIC_ID_PATH) \
		GENERATIVE_CHECKPOINT=$(ML32M_GENERATIVE_CHECKPOINT) \
		GENERATIVE_REPORT=$(ML32M_GENERATIVE_REPORT) \
		GENERATIVE_BATCH_SIZE=$(ML32M_GENERATIVE_BATCH_SIZE) \
		GENERATIVE_PARQUET_BATCH_SIZE=$(ML32M_GENERATIVE_PARQUET_BATCH_SIZE) \
		GENERATIVE_NUM_WORKERS=$(ML32M_GENERATIVE_NUM_WORKERS) \
		GENERATIVE_PREFETCH_FACTOR=$(ML32M_GENERATIVE_PREFETCH_FACTOR) \
		GENERATIVE_AMP=$(ML32M_GENERATIVE_AMP) \
		GENERATIVE_AMP_DTYPE=$(ML32M_GENERATIVE_AMP_DTYPE)
	$(MAKE) eval-generative-ranking \
		PROCESSED_DIR=$(ML32M_PROCESSED_DIR) \
		SEMANTIC_ID_PATH=$(ML32M_SEMANTIC_ID_PATH) \
		STATIC_DECODING_INDEX_PATH=$(ML32M_STATIC_DECODING_INDEX_PATH) \
		GENERATIVE_CHECKPOINT=$(ML32M_GENERATIVE_CHECKPOINT) \
		GENERATIVE_RANKING_REPORT=$(ML32M_GENERATIVE_RANKING_REPORT) \
		GENERATIVE_BEAM_SIZE=$(ML32M_GENERATIVE_BEAM_SIZE)

benchmark-ml32m:
	$(MAKE) benchmark-decoder \
		DECODER_BENCHMARK_SEMANTIC_ID_PATH=$(ML32M_SEMANTIC_ID_PATH) \
		DECODER_BENCHMARK_REPORT=$(ML32M_DECODER_BENCHMARK_REPORT)
	STATIC_REC_GENERATIVE_CHECKPOINT=$(ML32M_GENERATIVE_CHECKPOINT) \
	STATIC_REC_SEMANTIC_ID_PATH=$(ML32M_SEMANTIC_ID_PATH) \
	STATIC_REC_STATIC_DECODING_INDEX_PATH=$(ML32M_STATIC_DECODING_INDEX_PATH) \
	STATIC_REC_USER_HISTORY_PARQUET=$(ML32M_PROCESSED_DIR)/valid.parquet \
	STATIC_REC_DEVICE=$(ML32M_SERVING_DEVICE) \
	$(MAKE) benchmark-serving \
		SERVING_BENCHMARK_SERVICE=environment \
		SERVING_BENCHMARK_REPORT=$(ML32M_SERVING_BENCHMARK_REPORT)
	STATIC_REC_GENERATIVE_CHECKPOINT=$(ML32M_GENERATIVE_CHECKPOINT) \
	STATIC_REC_SEMANTIC_ID_PATH=$(ML32M_SEMANTIC_ID_PATH) \
	STATIC_REC_STATIC_DECODING_INDEX_PATH=$(ML32M_STATIC_DECODING_INDEX_PATH) \
	STATIC_REC_USER_HISTORY_PARQUET=$(ML32M_PROCESSED_DIR)/valid.parquet \
	STATIC_REC_DEVICE=$(ML32M_SERVING_DEVICE) \
	$(MAKE) benchmark-api \
		SERVING_BENCHMARK_SERVICE=environment \
		HTTP_SERVING_BENCHMARK_REPORT=$(ML32M_HTTP_SERVING_BENCHMARK_REPORT)

reproduce-ml32m: preprocess-ml32m baseline-ml32m semantic-ids-ml32m generative-ml32m benchmark-ml32m

serve-api:
	$(UV) run uvicorn apps.api.main:app --host $(API_HOST) --port $(API_PORT) --reload

check: lint test
