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
	@echo   student-dataset              Generate student degree dataset
	@echo   train-student-model          Train student classification model (custom NN)
	@echo   test-student-model           Test student classification model (custom NN)
	@echo   student-model               Complete student model pipeline (custom NN)
	@echo   train-tensorflow-student-model  Train TensorFlow/Keras student model
	@echo   test-tensorflow-student-model   Test TensorFlow/Keras student model
	@echo   tensorflow-student-model         Complete TensorFlow student model pipeline
	@echo   test-datasets                Test dataset loader
	@echo   lint                 Run linters
	@echo   format               Format code with black
	@echo   test                 Run all tests
	@echo   notebooks            Launch Jupyter notebooks
	@echo   clean                Clean generated files
	@echo   setup                Complete project setup
	@echo   all                  Run full setup and validation
	@echo   info                 Show project information
	@echo.
	@echo Mathematics and Visualization Commands:
	@echo   run-linear-algebra         Run linear algebra examples
	@echo   run-image-processing      Run image processing examples and generate visualizations
	@echo   generate-image-visualizations  Generate all image processing visualizations
	@echo   run-calculus              Run all calculus examples
	@echo   run-calculus-derivatives  Visualize function and its derivative
	@echo   run-calculus-tangents    Visualize tangent lines
	@echo   run-calculus-examples     Run calculus examples (integration, etc.)
	@echo   generate-calculus-visualizations  Generate all calculus visualizations
	@echo   run-gradients             Run all gradient examples
	@echo   run-gradients-examples    Run gradient examples (gradient descent, partial derivatives, etc.)
	@echo   run-gradients-field       Visualize gradient field (2D)
	@echo   run-gradients-descent     Visualize gradient descent optimization
	@echo   run-gradients-learning-rates  Compare different learning rates
	@echo   run-gradients-computation-graph  Visualize computation graph and gradient flow
	@echo   generate-gradients-visualizations  Generate all gradient visualizations
	@echo   run-statistics            Run statistics examples
	@echo   run-neural-networks       Run neural network examples
	@echo   run-cnn                  Run CNN examples
	@echo   run-rnn                  Run RNN examples
	@echo.
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

train-tensorflow-student-model: student-dataset ## Train TensorFlow/Keras student classification model
	@echo Training TensorFlow student classification model...
	@$(PYTHON) -c "import tensorflow" 2>nul || $(PIP) install -q tensorflow
	@$(PYTHON) -c "import sklearn; import numpy; import pandas" 2>nul || $(PIP) install -q scikit-learn numpy pandas
	$(PYTHON) $(SCRIPTS_DIR)/train_tensorflow_student_model.py
	@echo [OK] TensorFlow model training complete

test-tensorflow-student-model: ## Test TensorFlow/Keras student classification model
	@echo Testing TensorFlow student classification model...
	@$(PYTHON) -c "import tensorflow" 2>nul || $(PIP) install -q tensorflow
	@$(PYTHON) -c "import sklearn; import numpy; import pandas" 2>nul || $(PIP) install -q scikit-learn numpy pandas
	$(PYTHON) $(SCRIPTS_DIR)/test_tensorflow_student_model.py
	@echo [OK] TensorFlow model testing complete

tensorflow-student-model: train-tensorflow-student-model test-tensorflow-student-model ## Complete TensorFlow student model pipeline
	@echo [OK] TensorFlow student model pipeline complete

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
	@$(PYTHON) -c "import matplotlib; import numpy" 2>nul || $(PIP) install -q matplotlib numpy
	$(PYTHON) -c "from src.mathematics.linear_algebra import *; print('Linear algebra module loaded')"

run-image-processing: ## Run image processing examples and generate visualizations
	@echo Running image processing examples...
	@$(PYTHON) -c "import matplotlib; import numpy; import scipy" 2>nul || $(PIP) install -q matplotlib numpy scipy
	@echo Generating image processing visualizations...
	$(PYTHON) -c "from src.mathematics.image_processing import *; print('Image processing module loaded')"
	$(PYTHON) $(SRC_DIR)/mathematics/image_processing.py
	@echo [OK] Image processing visualizations generated in docs/images/

generate-image-visualizations: ## Generate all image processing visualizations
	@echo Generating image processing visualizations...
	@$(PYTHON) -c "import matplotlib; import numpy; import scipy" 2>nul || $(PIP) install -q matplotlib numpy scipy
	@if not exist docs\images mkdir docs\images
	$(PYTHON) -c "from src.mathematics.image_processing import visualize_image_as_matrix, visualize_image_transformations, visualize_image_filtering, visualize_image_compression_svd, visualize_edge_detection; visualize_image_as_matrix('docs/images/image_as_matrix.png'); visualize_image_transformations('docs/images/image_transformations_linear_algebra.png'); visualize_image_filtering('docs/images/image_filtering_convolution.png'); visualize_image_compression_svd('docs/images/image_compression_svd.png'); visualize_edge_detection('docs/images/edge_detection_matrix_operations.png'); print('[OK] All image processing visualizations generated')"
	@echo [OK] Image processing visualizations generated

run-calculus: ## Run all calculus examples
	@echo Running all calculus examples...
	@$(PYTHON) -c "import matplotlib; import numpy; import scipy" 2>nul || $(PIP) install -q matplotlib numpy scipy
	$(PYTHON) $(SRC_DIR)/mathematics/calculus.py

run-calculus-derivatives: ## Visualize function and its derivative
	@echo Generating function and derivative visualization...
	@$(PYTHON) -c "import matplotlib; import numpy" 2>nul || $(PIP) install -q matplotlib numpy
	@if not exist docs\images mkdir docs\images
	$(PYTHON) -c "from src.mathematics.calculus import visualize_function_and_derivative; import numpy as np; f = lambda x: x**3 - 3*x**2 + 2; df = lambda x: 3*x**2 - 6*x; visualize_function_and_derivative(f, df, save_path='docs/images/function_and_derivative.png'); print('[OK] Function and derivative visualization generated')"

run-calculus-tangents: ## Visualize tangent lines
	@echo Generating tangent lines visualization...
	@$(PYTHON) -c "import matplotlib; import numpy" 2>nul || $(PIP) install -q matplotlib numpy
	@if not exist docs\images mkdir docs\images
	$(PYTHON) -c "from src.mathematics.calculus import visualize_tangent_lines; import numpy as np; f = lambda x: x**2; df = lambda x: 2*x; visualize_tangent_lines(f, df, points=[-2, -1, 0, 1, 2], save_path='docs/images/tangent_lines.png'); print('[OK] Tangent lines visualization generated')"

# Gradient functions moved to gradients module - use run-gradients-* commands instead

run-gradients: ## Run all gradient examples
	@echo Running gradient examples...
	@$(PYTHON) -c "import matplotlib; import numpy" 2>nul || $(PIP) install -q matplotlib numpy
	$(PYTHON) $(SRC_DIR)/mathematics/gradients.py

run-gradients-examples: ## Run gradient examples (gradient descent, partial derivatives, neural network gradients)
	@echo Running gradient examples...
	@$(PYTHON) -c "import numpy" 2>nul || $(PIP) install -q numpy
	$(PYTHON) -c "from src.mathematics.gradients import gradient_descent_example, partial_derivatives_example, neural_network_gradient_example; print('=== Gradient Descent ==='); gradient_descent_example(); print('\n=== Partial Derivatives ==='); partial_derivatives_example(); print('\n=== Neural Network Gradients ==='); neural_network_gradient_example()"

run-gradients-field: ## Visualize gradient field (2D)
	@echo Generating gradient field visualization...
	@$(PYTHON) -c "import matplotlib; import numpy" 2>nul || $(PIP) install -q matplotlib numpy
	@if not exist docs\images mkdir docs\images
	$(PYTHON) -c "from src.mathematics.gradients import visualize_gradient_field; import numpy as np; f_2d = lambda x, y: x**2 + y**2; visualize_gradient_field(f_2d, save_path='docs/images/gradient_field.png'); print('[OK] Gradient field visualization generated')"

run-gradients-descent: ## Visualize gradient descent optimization
	@echo Generating gradient descent visualization...
	@$(PYTHON) -c "import matplotlib; import numpy" 2>nul || $(PIP) install -q matplotlib numpy
	@if not exist docs\images mkdir docs\images
	$(PYTHON) -c "from src.mathematics.gradients import visualize_gradient_descent; import numpy as np; f = lambda x: x**2 + 2*x + 1; df = lambda x: 2*x + 2; visualize_gradient_descent(f, df, x_start=5.0, learning_rate=0.1, iterations=50, save_path='docs/images/gradient_descent.png'); print('[OK] Gradient descent visualization generated')"

run-gradients-learning-rates: ## Compare different learning rates
	@echo Generating learning rate comparison visualization...
	@$(PYTHON) -c "import matplotlib; import numpy" 2>nul || $(PIP) install -q matplotlib numpy
	@if not exist docs\images mkdir docs\images
	$(PYTHON) -c "from src.mathematics.gradients import compare_learning_rates; import numpy as np; f = lambda x: x**2 + 2*x + 1; df = lambda x: 2*x + 2; compare_learning_rates(f, df, x_start=5.0, learning_rates=[0.01, 0.1, 0.5, 1.0], iterations=30, save_path='docs/images/learning_rates_comparison.png'); print('[OK] Learning rates comparison visualization generated')"

run-gradients-computation-graph: ## Visualize computation graph and gradient flow
	@echo Generating computation graph visualization...
	@$(PYTHON) -c "import matplotlib; import numpy" 2>nul || $(PIP) install -q matplotlib numpy
	@if not exist docs\images mkdir docs\images
	$(PYTHON) -c "from src.mathematics.gradients import visualize_computation_graph; visualize_computation_graph(save_path='docs/images/computation_graph.png'); print('[OK] Computation graph visualization generated')"

generate-gradients-visualizations: run-gradients-field run-gradients-descent run-gradients-learning-rates run-gradients-computation-graph ## Generate all gradient visualizations
	@echo [OK] All gradient visualizations generated

run-calculus-examples: ## Run calculus examples (integration, etc. - gradients moved to gradients module)
	@echo Running calculus examples...
	@$(PYTHON) -c "import numpy; import scipy" 2>nul || $(PIP) install -q numpy scipy
	$(PYTHON) -c "from src.mathematics.calculus import integration_example; print('=== Numerical Integration ==='); integration_example()"

generate-calculus-visualizations: run-calculus-derivatives run-calculus-tangents ## Generate all calculus visualizations
	@echo [OK] All calculus visualizations generated
	@echo Note: Gradient visualizations moved to generate-gradients-visualizations

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

run-student-classification: ## Run student degree classification example (custom NN)
	@echo Running student classification example...
	@$(PYTHON) -c "import matplotlib; import sklearn" 2>nul || $(PIP) install -q matplotlib scikit-learn
	$(PYTHON) $(SCRIPTS_DIR)/train_student_model.py

run-tensorflow-student-classification: ## Run TensorFlow student degree classification example
	@echo Running TensorFlow student classification example...
	@$(PYTHON) -c "import tensorflow" 2>nul || $(PIP) install -q tensorflow
	@$(PYTHON) -c "import sklearn; import numpy; import pandas" 2>nul || $(PIP) install -q scikit-learn numpy pandas
	$(PYTHON) $(SCRIPTS_DIR)/train_tensorflow_student_model.py

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

clean-models: ## Clean trained model files
	@echo Cleaning model files...
	@del /q $(DATA_DIR)\neural_networks\*.h5 2>nul
	@del /q $(DATA_DIR)\neural_networks\*_model_info.json 2>nul
	@del /q $(DATA_DIR)\neural_networks\training_history.csv 2>nul
	@echo [OK] Model files cleaned

clean-all: clean clean-datasets clean-models ## Clean everything including datasets and models
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
