.PHONY: help install install-dev format lint test clean dev run docker-build docker-run azure-setup

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install production dependencies
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev,test]"
	pre-commit install

format: ## Format code with black and ruff
	black src/ tests/ notebooks/
	ruff --fix src/ tests/ notebooks/

lint: ## Lint code with ruff and mypy
	ruff check src/ tests/ notebooks/
	mypy src/

test: ## Run tests
	pytest tests/ -v --cov=src --cov-report=html

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

dev: install-dev ## Set up development environment
	@echo "Development environment ready!"

run: ## Run the local development server
	cd src && python -m uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload

streamlit: ## Run the Streamlit dashboard
	streamlit run src/viz/dashboard_streamlit.py

simulate: ## Run IoT data simulator
	python src/data/simulate_stream.py

train: ## Train the autoencoder model
	python src/models/train_ae.py

export-onnx: ## Export trained model to ONNX
	python src/models/export_onnx.py

docker-build: ## Build Docker image
	docker build -t iot-anomaly:latest .

docker-run: ## Run Docker container
	docker run -p 8000:8000 iot-anomaly:latest

azure-setup: ## Create Azure resources (requires Azure CLI)
	@echo "Creating Azure resources..."
	@echo "Make sure you're logged in: az login"
	@echo "See infra/create_resources.md for detailed steps"
