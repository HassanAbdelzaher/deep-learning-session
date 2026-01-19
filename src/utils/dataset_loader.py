"""
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
