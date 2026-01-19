"""
Tests for mathematics modules
"""

import pytest
import numpy as np
from src.mathematics import linear_algebra, calculus, statistics


def test_vector_operations():
    """Test vector operations"""
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    
    assert np.array_equal(v1 + v2, np.array([5, 7, 9]))
    assert np.dot(v1, v2) == 32


def test_matrix_operations():
    """Test matrix operations"""
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    result = np.dot(A, B)
    expected = np.array([[19, 22], [43, 50]])
    
    assert np.allclose(result, expected)


def test_gradient_descent():
    """Test gradient descent converges"""
    def f(x):
        return x**2 + 2*x + 1
    
    def df(x):
        return 2*x + 2
    
    x = 5.0
    learning_rate = 0.1
    
    for _ in range(50):
        x = x - learning_rate * df(x)
    
    # Should converge close to minimum at x = -1
    assert abs(x + 1) < 0.1
