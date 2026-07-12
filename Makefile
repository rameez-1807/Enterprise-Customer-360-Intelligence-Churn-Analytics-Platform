# =============================================================================
# Makefile for Enterprise Customer 360 & Churn Analytics Platform
# =============================================================================

.PHONY: help install run pipeline test lint format docker-build docker-run clean

help:
	@echo "Enterprise Customer 360 Platform - Commands List:"
	@echo "  make install        Install requirements and setup virtual environment"
	@echo "  make run            Start Streamlit dashboard application"
	@echo "  make pipeline       Execute data cleaning, segmentation & model training"
	@echo "  make test           Run all pytest unit tests"
	@echo "  make lint           Check code quality with Flake8 & Mypy"
	@echo "  make format         Auto-format code using Black & Isort"
	@echo "  make docker-build   Build containerized application image"
	@echo "  make docker-run     Run containerized dashboard on port 8501"
	@echo "  make clean          Remove Python caching and log files"

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

run:
	streamlit run src/dashboard/app.py

pipeline:
	python main.py

test:
	python -m pytest --cov=src --cov-report=term-missing

lint:
	flake8 src tests
	mypy src tests

format:
	black src tests
	isort src tests

docker-build:
	docker build -t customer-360-platform:latest .

docker-run:
	docker run -d -p 8501:8501 --name customer_360_app customer-360-platform:latest

clean:
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete
	rm -f logs/*.log logs/*.json
