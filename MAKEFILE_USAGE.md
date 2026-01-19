# Makefile Usage Guide

This project includes a comprehensive Makefile for automating common tasks.

## Quick Start

```bash
# Show all available commands
make help

# Complete project setup
make setup

# Generate all datasets
make datasets

# Start Jupyter notebooks
make notebooks
```

## Available Commands

### Installation
- `make install` - Install project dependencies
- `make install-dev` - Install development dependencies (includes linting tools)
- `make setup` - Complete project setup (install + generate datasets)

### Dataset Management
- `make datasets` - Generate all datasets for neural network projects
- `make test-datasets` - Test dataset loader functionality

### Code Quality
- `make lint` - Run linters (flake8)
- `make format` - Format code with black
- `make type-check` - Run type checking with mypy

### Testing
- `make test` - Run all tests
- `make test-cov` - Run tests with coverage report
- `make validate` - Validate project (lint + test datasets)

### Notebooks
- `make notebooks` - Launch Jupyter Notebook server
- `make notebooks-lab` - Launch Jupyter Lab

### Cleaning
- `make clean` - Clean generated files and caches
- `make clean-datasets` - Remove generated datasets
- `make clean-all` - Clean everything including datasets

### Examples
- `make run-linear-algebra` - Run linear algebra examples
- `make run-calculus` - Run calculus examples
- `make run-statistics` - Run statistics examples
- `make run-neural-networks` - Run neural network examples
- `make run-cnn` - Run CNN examples
- `make run-rnn` - Run RNN examples

### Project Information
- `make info` - Show project information
- `make git-status` - Show git status
- `make git-add` - Stage all changes

### Development
- `make dev-setup` - Complete development setup
- `make pre-commit` - Run pre-commit checks

### All-in-One
- `make all` - Run full setup and validation

## Examples

### First Time Setup
```bash
# Install dependencies and generate datasets
make setup
```

### Daily Development
```bash
# Start Jupyter
make notebooks

# Before committing
make pre-commit
```

### Regenerating Datasets
```bash
# Clean old datasets
make clean-datasets

# Generate fresh datasets
make datasets

# Verify datasets work
make test-datasets
```

## Notes

- The Makefile is designed to work on both Windows and Unix systems
- Some commands require additional tools (black, flake8, pytest) - install with `make install-dev`
- Datasets are saved to `data/neural_networks/` directory
- All generated files can be cleaned with `make clean`
