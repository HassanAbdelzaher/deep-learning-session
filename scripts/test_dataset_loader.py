"""Test the dataset loader to ensure all datasets can be loaded correctly"""

import sys
from pathlib import Path
import importlib.util

# Load dataset_loader directly
spec = importlib.util.spec_from_file_location(
    "dataset_loader", 
    Path(__file__).parent.parent / "src" / "utils" / "dataset_loader.py"
)
dataset_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_loader)

import numpy as np

print("Testing Dataset Loader...")
print("=" * 60)

try:
    X, y = dataset_loader.load_spam()
    print(f"[OK] Spam dataset: {X.shape}, labels: {y.shape}")
except Exception as e:
    print(f"[ERROR] Spam: {e}")

try:
    X, y, m = dataset_loader.load_iris()
    print(f"[OK] Iris dataset: {X.shape}, labels: {y.shape}, classes: {m['n_classes']}")
except Exception as e:
    print(f"[ERROR] Iris: {e}")

try:
    X, y = dataset_loader.load_xor()
    print(f"[OK] XOR dataset: {X.shape}, labels: {y.shape}")
except Exception as e:
    print(f"[ERROR] XOR: {e}")

try:
    X, y, m = dataset_loader.load_house_prices()
    print(f"[OK] House prices dataset: {X.shape}, labels: {y.shape}, features: {len(m['feature_names'])}")
except Exception as e:
    print(f"[ERROR] House prices: {e}")

try:
    X, y = dataset_loader.load_cnn_images()
    print(f"[OK] CNN images dataset: {X.shape}, labels: {y.shape}")
except Exception as e:
    print(f"[ERROR] CNN images: {e}")

try:
    X, y = dataset_loader.load_mnist()
    print(f"[OK] MNIST dataset: {X.shape}, labels: {y.shape}")
except Exception as e:
    print(f"[ERROR] MNIST: {e}")

try:
    X, y_scores, y_categories, df = dataset_loader.load_student_degree()
    print(f"[OK] Student degree dataset: {X.shape}, scores: {y_scores.shape}, categories: {y_categories.shape}")
except Exception as e:
    print(f"[ERROR] Student degree: {e}")

print("=" * 60)
print("Dataset loader test complete!")
