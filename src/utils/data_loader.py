"""
Data loading utilities
"""

import numpy as np
import pandas as pd
from typing import Tuple


def load_synthetic_classification_data(n_samples=1000, n_features=20, n_classes=3, 
                                      random_state=42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic classification dataset
    
    Returns:
        X: Feature matrix
        y: Target labels
    """
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_features // 2,
        n_redundant=n_features // 4,
        random_state=random_state
    )
    
    return X, y


def load_synthetic_regression_data(n_samples=1000, n_features=10, 
                                   noise=0.1, random_state=42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression dataset
    
    Returns:
        X: Feature matrix
        y: Target values
    """
    from sklearn.datasets import make_regression
    
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state
    )
    
    return X, y


def normalize_data(X: np.ndarray, method='standard') -> Tuple[np.ndarray, dict]:
    """
    Normalize data
    
    Args:
        X: Data to normalize
        method: 'standard' or 'minmax'
    
    Returns:
        Normalized data and normalization parameters
    """
    if method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_norm = (X - mean) / (std + 1e-8)
        params = {'mean': mean, 'std': std}
    
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)
        params = {'min': min_val, 'max': max_val}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return X_norm, params
