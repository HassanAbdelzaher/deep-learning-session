"""
Generate and prepare datasets for applied neural network projects

This script generates/downloads all datasets required for the neural network projects
and saves them to the data/ directory for easy access.
"""

import numpy as np
import os
import pickle
from pathlib import Path

# Create data directory if it doesn't exist
data_dir = Path('data')
data_dir.mkdir(exist_ok=True)
neural_networks_dir = data_dir / 'neural_networks'
neural_networks_dir.mkdir(exist_ok=True)

print("=" * 60)
print("Generating Datasets for Neural Network Projects")
print("=" * 60)

# ============================================================================
# Project 1: MNIST Dataset
# ============================================================================
print("\n[1/6] Preparing MNIST Dataset...")
try:
    from sklearn.datasets import fetch_openml
    
    print("  Downloading MNIST from OpenML (this may take a few minutes)...")
    try:
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    except:
        # Fallback if liac-arff not available
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X_mnist = mnist.data.astype(np.float32) / 255.0  # Normalize to [0, 1]
    y_mnist = mnist.target.astype(int)
    
    # Save dataset
    np.save(neural_networks_dir / 'mnist_X.npy', X_mnist)
    np.save(neural_networks_dir / 'mnist_y.npy', y_mnist)
    
    print(f"  [OK] MNIST dataset saved: {X_mnist.shape[0]} samples, {X_mnist.shape[1]} features")
    print(f"    Classes: {len(np.unique(y_mnist))} (digits 0-9)")
except Exception as e:
    print(f"  [ERROR] Error downloading MNIST: {e}")
    print("    Note: MNIST will be downloaded automatically when running the notebook")

# ============================================================================
# Project 2: Spam Detection Dataset (Synthetic)
# ============================================================================
print("\n[2/6] Generating Spam Detection Dataset...")
try:
    from sklearn.datasets import make_classification
    
    X_spam, y_spam = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    y_spam = y_spam.reshape(-1, 1)
    
    # Save dataset
    np.save(neural_networks_dir / 'spam_X.npy', X_spam)
    np.save(neural_networks_dir / 'spam_y.npy', y_spam)
    
    print(f"  [OK] Spam dataset generated: {X_spam.shape[0]} samples, {X_spam.shape[1]} features")
    print(f"    Spam ratio: {np.mean(y_spam):.2%}")
except Exception as e:
    print(f"  [ERROR] Error generating spam dataset: {e}")

# ============================================================================
# Project 3: Iris Dataset
# ============================================================================
print("\n[3/6] Preparing Iris Dataset...")
try:
    from sklearn.datasets import load_iris
    
    iris = load_iris()
    X_iris = iris.data
    y_iris = iris.target
    
    # Save dataset
    np.save(neural_networks_dir / 'iris_X.npy', X_iris)
    np.save(neural_networks_dir / 'iris_y.npy', y_iris)
    
    # Save metadata
    metadata = {
        'feature_names': iris.feature_names,
        'target_names': iris.target_names.tolist(),
        'n_samples': len(X_iris),
        'n_features': X_iris.shape[1],
        'n_classes': len(iris.target_names)
    }
    with open(neural_networks_dir / 'iris_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"  [OK] Iris dataset saved: {X_iris.shape[0]} samples, {X_iris.shape[1]} features")
    print(f"    Classes: {iris.target_names.tolist()}")
except Exception as e:
    print(f"  [ERROR] Error preparing Iris dataset: {e}")

# ============================================================================
# Project 4: House Price Prediction Dataset (Synthetic)
# ============================================================================
print("\n[4/6] Generating House Price Prediction Dataset...")
try:
    from sklearn.datasets import make_regression
    
    X_house, y_house = make_regression(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        noise=20,
        random_state=42
    )
    y_house = y_house.reshape(-1, 1)
    
    # Save dataset
    np.save(neural_networks_dir / 'house_X.npy', X_house)
    np.save(neural_networks_dir / 'house_y.npy', y_house)
    
    # Create feature names
    feature_names = [
        'Size (sqft)', 'Bedrooms', 'Bathrooms', 'Age (years)',
        'Location_Score', 'School_Rating', 'Crime_Rate',
        'Distance_to_City', 'Public_Transport', 'Parking_Spaces'
    ]
    metadata = {
        'feature_names': feature_names,
        'target_name': 'Price',
        'n_samples': len(X_house),
        'n_features': X_house.shape[1]
    }
    with open(neural_networks_dir / 'house_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"  [OK] House price dataset generated: {X_house.shape[0]} samples, {X_house.shape[1]} features")
    print(f"    Price range: ${y_house.min():.0f} - ${y_house.max():.0f}")
except Exception as e:
    print(f"  [ERROR] Error generating house price dataset: {e}")

# ============================================================================
# Project 5: XOR Dataset
# ============================================================================
print("\n[5/6] Preparing XOR Dataset...")
try:
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y_xor = np.array([[0], [1], [1], [0]], dtype=np.float32)
    
    # Save dataset
    np.save(neural_networks_dir / 'xor_X.npy', X_xor)
    np.save(neural_networks_dir / 'xor_y.npy', y_xor)
    
    print(f"  [OK] XOR dataset saved: {X_xor.shape[0]} samples, {X_xor.shape[1]} features")
    print("    This is the classic XOR problem dataset")
except Exception as e:
    print(f"  [ERROR] Error preparing XOR dataset: {e}")

# ============================================================================
# Project 6: CNN Image Dataset (Synthetic)
# ============================================================================
print("\n[6/6] Generating CNN Image Classification Dataset...")
try:
    np.random.seed(42)
    num_samples = 2000
    num_classes = 10
    
    # Generate synthetic 32x32 RGB images
    X_cnn = np.random.rand(num_samples, 3, 32, 32).astype(np.float32)
    y_cnn = np.random.randint(0, num_classes, num_samples)
    
    # Save dataset
    np.save(neural_networks_dir / 'cnn_X.npy', X_cnn)
    np.save(neural_networks_dir / 'cnn_y.npy', y_cnn)
    
    print(f"  [OK] CNN image dataset generated: {X_cnn.shape[0]} samples")
    print(f"    Image shape: {X_cnn.shape[1:]} (RGB, 32x32)")
    print(f"    Classes: {num_classes}")
except Exception as e:
    print(f"  [ERROR] Error generating CNN dataset: {e}")

# ============================================================================
# Create Dataset Loader Utility
# ============================================================================
print("\n" + "=" * 60)
print("Creating Dataset Loader Utility...")
print("=" * 60)

loader_code = '''"""
Dataset Loader for Neural Network Projects

This module provides convenient functions to load pre-generated datasets.
"""

import numpy as np
import pickle
from pathlib import Path

DATA_DIR = Path('data/neural_networks')

def load_mnist():
    """Load MNIST dataset"""
    X = np.load(DATA_DIR / 'mnist_X.npy')
    y = np.load(DATA_DIR / 'mnist_y.npy')
    return X, y

def load_spam():
    """Load spam detection dataset"""
    X = np.load(DATA_DIR / 'spam_X.npy')
    y = np.load(DATA_DIR / 'spam_y.npy')
    return X, y

def load_iris():
    """Load Iris dataset with metadata"""
    X = np.load(DATA_DIR / 'iris_X.npy')
    y = np.load(DATA_DIR / 'iris_y.npy')
    with open(DATA_DIR / 'iris_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    return X, y, metadata

def load_house_prices():
    """Load house price prediction dataset with metadata"""
    X = np.load(DATA_DIR / 'house_X.npy')
    y = np.load(DATA_DIR / 'house_y.npy')
    with open(DATA_DIR / 'house_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    return X, y, metadata

def load_xor():
    """Load XOR dataset"""
    X = np.load(DATA_DIR / 'xor_X.npy')
    y = np.load(DATA_DIR / 'xor_y.npy')
    return X, y

def load_cnn_images():
    """Load CNN image classification dataset"""
    X = np.load(DATA_DIR / 'cnn_X.npy')
    y = np.load(DATA_DIR / 'cnn_y.npy')
    return X, y
'''

loader_path = Path('src/utils/dataset_loader.py')
loader_path.parent.mkdir(parents=True, exist_ok=True)
with open(loader_path, 'w') as f:
    f.write(loader_code)

print("  [OK] Dataset loader created at: src/utils/dataset_loader.py")

# ============================================================================
# Summary
# ============================================================================
# ============================================================================
# Project 7: Student Degree Classification Dataset
# ============================================================================
print("\n[7/7] Generating Student Degree Classification Dataset...")
try:
    import pandas as pd
    from scripts.generate_student_dataset import *
    # Run the student dataset generator
    exec(open('scripts/generate_student_dataset.py').read())
    print("  [OK] Student degree dataset generated")
except Exception as e:
    print(f"  [ERROR] Error generating student dataset: {e}")
    print("    Run 'python scripts/generate_student_dataset.py' separately")

print("\n" + "=" * 60)
print("Dataset Generation Complete!")
print("=" * 60)
print(f"\nAll datasets saved to: {neural_networks_dir}")
print("\nGenerated datasets:")
print("  1. MNIST - Handwritten digit recognition")
print("  2. Spam Detection - Binary classification")
print("  3. Iris - Multi-class classification")
print("  4. House Prices - Regression")
print("  5. XOR - Non-linearity demonstration")
print("  6. CNN Images - Image classification")
print("  7. Student Degree - CSV-based classification")
print("\nYou can now use these datasets in your notebooks!")
print("\nExample usage:")
print("  from src.utils.dataset_loader import load_mnist, load_student_degree")
print("  X, y = load_mnist()")
print("  X, scores, categories, df = load_student_degree()")