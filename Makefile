.PHONY: setup download simulate test lint clean

# ── Setup ─────────────────────────────────────────────────────────────────────
setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

# ── Data ──────────────────────────────────────────────────────────────────────
download:
	python scripts/download_datasets.py --dataset all

download-hydraulic:
	python scripts/download_datasets.py --dataset hydraulic

download-cmapss:
	python scripts/download_datasets.py --dataset cmapss

download-bearing:
	python scripts/download_datasets.py --dataset bearing

simulate:
	python simulator/sensor_generator.py --mode batch --units 200 --output data/simulated/

# ── Dev ───────────────────────────────────────────────────────────────────────
api:
	uvicorn api.main:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py --server.port 8501

mlflow:
	mlflow ui --port 5000

# ── Quality ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=. --cov-report=term-missing

lint:
	ruff check . --fix
	ruff format .

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker build -t mineguard:latest .

docker-up:
	docker-compose up --build

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage
