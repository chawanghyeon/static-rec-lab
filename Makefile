UV ?= uv
PYTHON_TARGETS := apps recsys scripts tests

.PHONY: format lint test benchmark-decoder check

format:
	$(UV) run ruff format $(PYTHON_TARGETS)
	$(UV) run ruff check --fix $(PYTHON_TARGETS)

lint:
	$(UV) run ruff format --check $(PYTHON_TARGETS)
	$(UV) run ruff check $(PYTHON_TARGETS)
	$(UV) run mypy

test:
	$(UV) run pytest

benchmark-decoder:
	$(UV) run python scripts/benchmark_decoder.py

check: lint test
