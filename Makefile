.PHONY: verify test lint typecheck

UV_CACHE_DIR ?= .uv-cache

verify: test lint typecheck

test:
	uv --cache-dir $(UV_CACHE_DIR) run pytest -q

lint:
	uv --cache-dir $(UV_CACHE_DIR) run ruff check .

typecheck:
	uv --cache-dir $(UV_CACHE_DIR) run mypy
