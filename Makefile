PY ?= python3

.PHONY: lint format

lint:
	@echo "Running Ruff lint..."
	@ruff check .

format:
	@echo "Formatting with Black..."
	@black --line-length 100 .

