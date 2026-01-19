# Makefile for Python AI Learning Project
# Comprehensive automation for dataset generation, testing, and project management

.PHONY: help install install-dev clean test lint format datasets notebooks docs all

# Default Python interpreter
PYTHON := python
PIP := pip

# Directories
SRC_DIR := src
NOTEBOOKS_DIR := notebooks
DATA_DIR := data
DOCS_DIR := docs
SCRIPTS_DIR := scripts
TESTS_DIR := tests

# Colors for output (removed for Windows compatibility)
# BLUE := \033[0;34m
# GREEN := \033[0;32m
# YELLOW := \033[0;33m
# RED := \033[0;31m
# NC := \033[0m # No Color

help: ## Show this help message
	@echo Python AI Learning Project - Makefile Commands
	@echo
	@echo Available commands:
	@echo   install              Install project dependencies
	@echo   install-dev          Install development dependencies
	@echo   datasets             Generate all datasets
	@echo   student-dataset      Generate student degree dataset
	@echo   train-student-model  Train student classification model
	@echo   test-student-model   Test student classification model
	@echo   student-model        Complete student model pipeline
	@echo   test-datasets        Test dataset loader
	@echo   lint                 Run linters
	@echo   format               Format code with black
	@echo   test                 Run all tests
	@echo   notebooks            Launch Jupyter notebooks
	@echo   clean                Clean generated files
	@echo   setup                Complete project setup
	@echo   all                  Run full setup and validation
	@echo   info                 Show project information
	@echo
	@echo For more commands, see the Makefile

# ============================================================================
# Installation
# ============================================================================

install: ## Install project dependencies
	@echo Installing dependencies...
	$(PIP) install -r requirements.txt
	@echo [OK] Dependencies installed

install-dev: install ## Install development dependencies
	@echo Installing development dependencies...
	$(PIP) install -r requirements.txt
	$(PIP) install black flake8 pytest pytest-cov mypy
	@echo [OK] Development dependencies installed

# ============================================================================
# Dataset Generation
# ============================================================================

datasets: ## Generate all datasets for neural network projects
	@echo Generating datasets...
	$(PYTHON) $(SCRIPTS_DIR)/generate_neural_network_datasets.py
	@echo Generating student degree dataset...
	$(PYTHON) $(SCRIPTS_DIR)/generate_student_dataset.py
	@echo [OK] Datasets generated

student-dataset: ## Generate student degree classification dataset
	@echo Generating student degree dataset...
	$(PYTHON) $(SCRIPTS_DIR)/generate_student_dataset.py
	@echo [OK] Student dataset generated

train-student-model: student-dataset ## Train student degree classification model
	@echo Training student classification model...
	@$(PYTHON) -c "import matplotlib; import sklearn" 2>nul || $(PIP) install -q matplotlib scikit-learn
	$(PYTHON) $(SCRIPTS_DIR)/train_student_model.py
	@echo [OK] Model training complete

test-student-model: ## Test student degree classification model
	@echo Testing student classification model...
	@$(PYTHON) -c "import matplotlib; import sklearn" 2>nul || $(PIP) install -q matplotlib scikit-learn
	$(PYTHON) $(SCRIPTS_DIR)/test_student_model.py
	@echo [OK] Model testing complete

student-model: train-student-model test-student-model ## Complete student model pipeline (dataset + train + test)
	@echo [OK] Student model pipeline complete

test-datasets: ## Test dataset loader functionality
	@echo Testing dataset loader...
	$(PYTHON) $(SCRIPTS_DIR)/test_dataset_loader.py
	@echo [OK] Dataset loader test complete

# ============================================================================
# Code Quality
# ============================================================================

lint: ## Run linters (flake8)
	@echo Running linters...
	@where flake8 >nul 2>&1 && flake8 $(SRC_DIR) --max-line-length=100 --ignore=E203,W503 || echo flake8 not installed. Run 'make install-dev'
	@echo [OK] Linting complete

format: ## Format code with black
	@echo Formatting code...
	@where black >nul 2>&1 && black $(SRC_DIR) $(SCRIPTS_DIR) --line-length=100 || echo black not installed. Run 'make install-dev'
	@echo [OK] Code formatted

type-check: ## Run type checking with mypy
	@echo Running type checks...
	@where mypy >nul 2>&1 && mypy $(SRC_DIR) --ignore-missing-imports || echo mypy not installed. Run 'make install-dev'
	@echo [OK] Type checking complete

# ============================================================================
# Testing
# ============================================================================

test: ## Run all tests
	@echo Running tests...
	@where pytest >nul 2>&1 && pytest $(TESTS_DIR) -v || echo pytest not installed. Run 'make install-dev'
	@echo [OK] Tests complete

test-cov: ## Run tests with coverage report
	@echo Running tests with coverage...
	@where pytest >nul 2>&1 && pytest $(TESTS_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term || echo pytest not installed. Run 'make install-dev'
	@echo [OK] Coverage report generated

# ============================================================================
# Notebooks
# ============================================================================

notebooks: ## Launch Jupyter notebooks
	@echo Starting Jupyter Notebook server...
	@where jupyter >nul 2>&1 && jupyter notebook || echo Jupyter not installed. Run 'make install'

notebooks-lab: ## Launch Jupyter Lab
	@echo Starting Jupyter Lab...
	@where jupyter >nul 2>&1 && jupyter lab || echo Jupyter not installed. Run 'make install'

# ============================================================================
# Documentation
# ============================================================================

docs: ## Generate documentation (if using Sphinx)
	@echo Generating documentation...
	@if exist docs\_build (cd docs && make html) else (echo Documentation is in markdown format in docs/ directory && echo [OK] View documentation at: docs/README.md)

# ============================================================================
# Examples and Demos
# ============================================================================

run-linear-algebra: ## Run linear algebra examples
	@echo Running linear algebra examples...
	$(PYTHON) -c "from src.mathematics.linear_algebra import *; print('Linear algebra module loaded')"

run-calculus: ## Run calculus examples
	@echo Running calculus examples...
	$(PYTHON) -c "from src.mathematics.calculus import *; print('Calculus module loaded')"

run-statistics: ## Run statistics examples
	@echo Running statistics examples...
	$(PYTHON) -c "from src.mathematics.statistics import *; print('Statistics module loaded')"

run-neural-networks: ## Run neural network examples
	@echo Running neural network examples...
	$(PYTHON) -c "from src.deep_learning.neural_networks import *; print('Neural networks module loaded')"

run-cnn: ## Run CNN examples
	@echo Running CNN examples...
	$(PYTHON) -c "from src.deep_learning.cnn import *; print('CNN module loaded')"

run-rnn: ## Run RNN examples
	@echo Running RNN examples...
	$(PYTHON) -c "from src.deep_learning.rnn import *; print('RNN module loaded')"

run-student-classification: ## Run student degree classification example
	@echo Running student classification example...
	$(PYTHON) $(SCRIPTS_DIR)/train_student_model.py

# ============================================================================
# Project Management
# ============================================================================

clean: ## Clean generated files and caches
	@echo Cleaning project...
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
	@for /r . %%f in (*.pyc) do @del /q "%%f" 2>nul
	@for /r . %%f in (*.pyo) do @del /q "%%f" 2>nul
	@for /d /r . %%d in (*.egg-info) do @if exist "%%d" rd /s /q "%%d" 2>nul
	@for /d /r . %%d in (.pytest_cache) do @if exist "%%d" rd /s /q "%%d" 2>nul
	@for /d /r . %%d in (.mypy_cache) do @if exist "%%d" rd /s /q "%%d" 2>nul
	@for /d /r . %%d in (.ipynb_checkpoints) do @if exist "%%d" rd /s /q "%%d" 2>nul
	@echo [OK] Cleanup complete

clean-datasets: ## Remove generated datasets (keeps structure)
	@echo Cleaning datasets...
	@del /q $(DATA_DIR)\neural_networks\*.npy 2>nul
	@del /q $(DATA_DIR)\neural_networks\*.pkl 2>nul
	@echo [OK] Datasets cleaned (run 'make datasets' to regenerate)

clean-all: clean clean-datasets ## Clean everything including datasets
	@echo [OK] Full cleanup complete

# ============================================================================
# Quick Start
# ============================================================================

setup: install datasets ## Complete project setup (install + generate datasets)
	@echo [OK] Project setup complete!
	@echo
	@echo Next steps:
	@echo   1. Run 'make notebooks' to start Jupyter
	@echo   2. Explore notebooks in $(NOTEBOOKS_DIR)/
	@echo   3. Read documentation in $(DOCS_DIR)/

# ============================================================================
# Validation
# ============================================================================

validate: lint test-datasets ## Validate project (lint + test datasets)
	@echo [OK] Project validation complete

# ============================================================================
# All-in-one
# ============================================================================

all: clean install datasets test-datasets ## Run full setup and validation
	@echo [OK] All tasks complete!

# ============================================================================
# Project Information
# ============================================================================

info: ## Show project information
	@echo Python AI Learning Project
	@echo
	@echo Project Structure:
	@echo   Source Code:     $(SRC_DIR)/
	@echo   Notebooks:       $(NOTEBOOKS_DIR)/
	@echo   Documentation:   $(DOCS_DIR)/
	@echo   Datasets:        $(DATA_DIR)/
	@echo   Scripts:         $(SCRIPTS_DIR)/
	@echo   Tests:           $(TESTS_DIR)/
	@echo
	@echo Python Version:
	@$(PYTHON) --version
	@echo
	@echo Installed Packages:
	@$(PIP) list | findstr /C:"numpy" /C:"matplotlib" /C:"scikit-learn" /C:"torch" /C:"jupyter" || echo   Run 'make install' to install packages

# ============================================================================
# Git Operations
# ============================================================================

git-status: ## Show git status
	@echo Git Status:
	@git status

git-add: ## Stage all changes
	@echo Staging all changes...
	@git add .
	@echo [OK] Changes staged

# ============================================================================
# Development Workflow
# ============================================================================

dev-setup: install-dev datasets test-datasets ## Complete development setup
	@echo [OK] Development environment ready!

pre-commit: format lint test-datasets ## Run pre-commit checks
	@echo [OK] Pre-commit checks passed

# ============================================================================
# Docker Support (if needed)
# ============================================================================

docker-build: ## Build Docker image (if Dockerfile exists)
	@if exist Dockerfile (echo Building Docker image... && docker build -t py-ai-learning . && echo [OK] Docker image built) else (echo No Dockerfile found)

docker-run: ## Run Docker container
	@if exist Dockerfile (echo Running Docker container... && docker run -it -p 8888:8888 -v %cd%:/workspace py-ai-learning) else (echo No Dockerfile found)
