PY ?= python3

.PHONY: lint format

lint:
	@echo "Running Ruff lint..."
	@ruff check scripts

format:
	@echo "Formatting with Ruff..."
	@ruff format scripts

