"""
Linear Algebra Fundamentals
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


def vector_operations_example():
    """Demonstrate basic vector operations"""
    # Create vectors
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    
    # Vector addition
    v_sum = v1 + v2
    print(f"Vector addition: {v1} + {v2} = {v_sum}")
    
    # Scalar multiplication
    scalar = 2
    v_scaled = scalar * v1
    print(f"Scalar multiplication: {scalar} * {v1} = {v_scaled}")
    
    # Dot product
    dot_product = np.dot(v1, v2)
    print(f"Dot product: {v1} · {v2} = {dot_product}")
    
    # Cross product (3D only)
    if len(v1) == 3:
        cross_product = np.cross(v1, v2)
        print(f"Cross product: {v1} × {v2} = {cross_product}")
    
    return v1, v2, v_sum, dot_product


def matrix_operations_example():
    """Demonstrate basic matrix operations"""
    # Create matrices
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    # Matrix addition
    M_sum = A + B
    print(f"Matrix addition:\n{A}\n+\n{B}\n=\n{M_sum}\n")
    
    # Matrix multiplication
    M_product = np.dot(A, B)
    print(f"Matrix multiplication:\n{A}\n×\n{B}\n=\n{M_product}\n")
    
    # Matrix transpose
    A_transpose = A.T
    print(f"Matrix transpose:\n{A}\n^T\n=\n{A_transpose}\n")
    
    # Matrix determinant
    det_A = np.linalg.det(A)
    print(f"Determinant of A: {det_A}")
    
    # Matrix inverse
    if det_A != 0:
        A_inverse = np.linalg.inv(A)
        print(f"Inverse of A:\n{A_inverse}\n")
    
    return A, B, M_product, det_A


def eigenvalues_eigenvectors_example():
    """Demonstrate eigenvalues and eigenvectors"""
    # Create a symmetric matrix
    A = np.array([[4, 2], [2, 3]])
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print(f"Matrix A:\n{A}\n")
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}\n")
    
    # Verify: Av = λv
    for i, (eigenvalue, eigenvector) in enumerate(zip(eigenvalues, eigenvectors.T)):
        result = np.dot(A, eigenvector)
        expected = eigenvalue * eigenvector
        print(f"Verification {i+1}: Av = {result}, λv = {expected}")
        print(f"Match: {np.allclose(result, expected)}\n")
    
    return eigenvalues, eigenvectors


def visualize_vectors_2d(vectors: List[np.ndarray], labels: List[str] = None):
    """Visualize 2D vectors"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for i, vec in enumerate(vectors):
        label = labels[i] if labels else f"Vector {i+1}"
        ax.arrow(0, 0, vec[0], vec[1], head_width=0.1, head_length=0.1,
                fc='blue', ec='blue', label=label)
        ax.text(vec[0]*1.1, vec[1]*1.1, label, fontsize=10)
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Vector Visualization')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend()
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("=== Vector Operations ===")
    vector_operations_example()
    
    print("\n=== Matrix Operations ===")
    matrix_operations_example()
    
    print("\n=== Eigenvalues and Eigenvectors ===")
    eigenvalues_eigenvectors_example()
