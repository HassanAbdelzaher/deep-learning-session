# Neural Network Datasets

This directory contains pre-generated datasets for the applied neural network projects.

## Generated Datasets

### 1. MNIST (`mnist_X.npy`, `mnist_y.npy`)
- **Project**: Handwritten Digit Recognition
- **Samples**: 70,000
- **Features**: 784 (28×28 flattened images)
- **Classes**: 10 (digits 0-9)
- **Note**: If not present, will be downloaded automatically when running notebooks

### 2. Spam Detection (`spam_X.npy`, `spam_y.npy`)
- **Project**: Binary Classification - Spam Detection
- **Samples**: 5,000
- **Features**: 20 (email characteristics)
- **Classes**: 2 (spam/not spam)

### 3. Iris (`iris_X.npy`, `iris_y.npy`, `iris_metadata.pkl`)
- **Project**: Multi-class Classification - Iris Species
- **Samples**: 150
- **Features**: 4 (sepal length, sepal width, petal length, petal width)
- **Classes**: 3 (setosa, versicolor, virginica)
- **Metadata**: Contains feature names and target names

### 4. House Prices (`house_X.npy`, `house_y.npy`, `house_metadata.pkl`)
- **Project**: Regression - House Price Prediction
- **Samples**: 1,000
- **Features**: 10 (size, bedrooms, bathrooms, age, location, etc.)
- **Target**: Continuous (house price)
- **Metadata**: Contains feature names

### 5. XOR (`xor_X.npy`, `xor_y.npy`)
- **Project**: XOR Problem - Non-linearity Demonstration
- **Samples**: 4
- **Features**: 2
- **Classes**: 2
- **Note**: Classic XOR problem dataset

### 6. CNN Images (`cnn_X.npy`, `cnn_y.npy`)
- **Project**: Image Classification with CNN
- **Samples**: 2,000
- **Image Shape**: (3, 32, 32) - RGB images
- **Classes**: 10

## Usage

### Option 1: Use Dataset Loader (Recommended)

```python
from src.utils.dataset_loader import (
    load_mnist,
    load_spam,
    load_iris,
    load_house_prices,
    load_xor,
    load_cnn_images
)

# Load datasets
X, y = load_mnist()
X, y = load_spam()
X, y, metadata = load_iris()  # Includes metadata
X, y, metadata = load_house_prices()  # Includes metadata
X, y = load_xor()
X, y = load_cnn_images()
```

### Option 2: Load Directly

```python
import numpy as np
import pickle
from pathlib import Path

data_dir = Path('data/neural_networks')

# Load numpy arrays
X = np.load(data_dir / 'spam_X.npy')
y = np.load(data_dir / 'spam_y.npy')

# Load metadata (for Iris and House Prices)
with open(data_dir / 'iris_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
```

## Regenerating Datasets

To regenerate all datasets, run:

```bash
python scripts/generate_neural_network_datasets.py
```

This will:
- Download/generate all datasets
- Save them to this directory
- Create/update the dataset loader utility

## File Sizes

- **MNIST**: ~50-60 MB (if downloaded)
- **Spam**: ~800 KB
- **Iris**: ~5 KB
- **House Prices**: ~80 KB
- **XOR**: ~160 bytes
- **CNN Images**: ~24 MB

## Notes

- MNIST dataset is large and may take time to download on first run
- All datasets are saved in NumPy format (.npy) for fast loading
- Metadata files (.pkl) contain additional information like feature names
- Datasets are generated with fixed random seeds for reproducibility
