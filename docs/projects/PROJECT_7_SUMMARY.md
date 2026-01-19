# Project 7: Student Degree Classification - Summary

## Overview

This project implements a complete student degree classification system using a simple neural network with CSV dataset processing.

## Dataset

**File**: `data/neural_networks/student_degree_dataset.csv`

**Features** (7):
1. `attendance` - Class attendance percentage (60-100%)
2. `quiz_avg` - Average quiz score (40-95)
3. `assignment_avg` - Average assignment score (45-98)
4. `midterm_score` - Midterm exam score (35-100)
5. `project_score` - Project score (50-100)
6. `study_hours_per_week` - Study hours (5-40)
7. `participation_score` - Participation score (30-100)

**Target**: `degree_category` (5 classes)
- Bad: score < 50
- Acceptable: 50 ≤ score < 65
- Good: 65 ≤ score < 75
- Very Good: 75 ≤ score < 85
- Excellent: score ≥ 85

**Dataset Size**: 1,000 students

## Neural Network Architecture

- **Input Layer**: 7 neurons (one per feature)
- **Hidden Layer 1**: 32 neurons
- **Hidden Layer 2**: 16 neurons
- **Output Layer**: 5 neurons (one per degree category)

## Key Features

1. **CSV Data Processing**: Loads data from CSV file
2. **Feature Scaling**: StandardScaler for normalization
3. **Multi-class Classification**: 5 degree categories
4. **One-hot Encoding**: Converts categories to numerical format
5. **Model Evaluation**: Confusion matrix and classification report
6. **Feature Analysis**: Statistics by degree category
7. **Prediction**: Can predict on new student data

## Usage

### Generate Dataset
```bash
python scripts/generate_student_dataset.py
```

### Load Dataset
```python
from src.utils.dataset_loader import load_student_degree
X, y_scores, y_categories, df = load_student_degree()
```

### Run Notebook
Open `notebooks/09_project_7_student_degree_classification.ipynb` in Jupyter

## Learning Outcomes

- CSV data loading and preprocessing
- Multi-class classification with 5 classes
- Feature engineering and scaling
- Real-world educational data application
- Model evaluation and interpretation
- Feature importance analysis

## Requirements

- pandas (for CSV handling)
- numpy
- scikit-learn
- matplotlib
- SimpleNeuralNetwork from src.deep_learning.neural_networks
