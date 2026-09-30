.PHONY: install test lint health benchmark

install:
	python -m pip install -e '.[dev,plot]'

test:
	pytest -q

lint:
	ruff check .

health:
	vllm-bench health

benchmark:
	vllm-bench run --requests 50 --concurrency 4
