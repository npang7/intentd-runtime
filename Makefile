# GNU make shortcuts; the same Python commands work directly on Windows.
.PHONY: check lint types test chaos demo recovery clean

check: lint types test

lint:
	python -m ruff check .
	python -m ruff format --check .

types:
	python -m mypy

test:
	python -m pytest

chaos:
	python -m bench.chaos --n 150

demo:
	python -m orchestrator.run --scenario coverage_hole_01 --model oracle:standard --db runs/demo.db --run-id demo
recovery:
	python -m examples.recovery_demo

clean:
	python -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in (Path('.mypy_cache'), Path('.ruff_cache'), Path('.pytest_cache'))]"
