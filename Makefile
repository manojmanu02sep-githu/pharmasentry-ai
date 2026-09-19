.PHONY: setup seed run test lint typecheck eval docker-up docker-down

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@if [ ! -f .env ]; then cp .env.example .env; fi

seed:
	$(PYTHON) scripts/seed_synthetic_data.py

run:
	$(VENV)/bin/streamlit run app.py

test:
	$(VENV)/bin/pytest -v

lint:
	$(VENV)/bin/ruff check .

typecheck:
	$(VENV)/bin/mypy src config

eval:
	$(PYTHON) scripts/run_golden_evaluation.py

docker-up:
	docker compose up --build

docker-down:
	docker compose down
