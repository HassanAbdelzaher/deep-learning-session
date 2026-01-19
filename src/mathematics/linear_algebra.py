"""
Linear Algebra Fundamentals
Comprehensive module with visualizations and examples
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
try:
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    Axes3D = None


# ============================================================================
# VECTOR OPERATIONS
# ============================================================================

def vector_operations_example():
    """Demonstrate basic vector operations"""
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    
    v_sum = v1 + v2
    print(f"Vector addition: {v1} + {v2} = {v_sum}")
    
    scalar = 2
    v_scaled = scalar * v1
    print(f"Scalar multiplication: {scalar} * {v1} = {v_scaled}")
    
    dot_product = np.dot(v1, v2)
    print(f"Dot product: {v1} · {v2} = {dot_product}")
    
    if len(v1) == 3:
        cross_product = np.cross(v1, v2)
        print(f"Cross product: {v1} × {v2} = {cross_product}")
    
    return v1, v2, v_sum, dot_product


def visualize_vector_addition(v1: np.ndarray, v2: np.ndarray, save_path: Optional[str] = None):
    """Visualize vector addition with parallelogram rule"""
    v_sum = v1 + v2
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.arrow(0, 0, v1[0], v1[1], head_width=0.1, head_length=0.1, 
             fc='blue', ec='blue', label='v1', linewidth=2)
    ax.arrow(0, 0, v2[0], v2[1], head_width=0.1, head_length=0.1, 
             fc='green', ec='green', label='v2', linewidth=2)
    ax.arrow(0, 0, v_sum[0], v_sum[1], head_width=0.1, head_length=0.1, 
             fc='red', ec='red', label='v1 + v2', linewidth=2)
    ax.arrow(v1[0], v1[1], v2[0], v2[1], head_width=0.1, head_length=0.1, 
             fc='orange', ec='orange', linestyle='--', alpha=0.5, linewidth=1)
    
    max_val = max(v_sum.max(), v1.max(), v2.max())
    ax.set_xlim(-1, max(5, max_val + 1))
    ax.set_ylim(-1, max(5, max_val + 1))
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Vector Addition', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def visualize_dot_product(v1: np.ndarray, v2: np.ndarray, save_path: Optional[str] = None):
    """Visualize dot product with angle and projection"""
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    angle_rad = np.arccos(np.clip(dot_product / (magnitude_v1 * magnitude_v2), -1, 1))
    angle_deg = np.degrees(angle_rad)
    proj_v2_on_v1 = (dot_product / (magnitude_v1**2)) * v1
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].arrow(0, 0, v1[0], v1[1], head_width=0.2, head_length=0.2,
                 fc='blue', ec='blue', linewidth=3, label=f'v1 = {v1}')
    axes[0].arrow(0, 0, v2[0], v2[1], head_width=0.2, head_length=0.2,
                 fc='red', ec='red', linewidth=3, label=f'v2 = {v2}')
    
    theta = np.linspace(0, angle_rad, 100)
    r = 0.8
    axes[0].plot(r * np.cos(theta), r * np.sin(theta), 'g-', linewidth=2)
    axes[0].text(0.5, 0.3, f'θ = {angle_deg:.1f}°', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    max_val = max(abs(v1).max(), abs(v2).max())
    axes[0].set_xlim(-1, max(4, max_val + 1))
    axes[0].set_ylim(-1, max(4, max_val + 1))
    axes[0].set_xlabel('X', fontsize=12)
    axes[0].set_ylabel('Y', fontsize=12)
    axes[0].set_title('Dot Product: Angle Between Vectors', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linewidth=0.5)
    axes[0].axvline(x=0, color='k', linewidth=0.5)
    axes[0].legend(fontsize=10)
    axes[0].set_aspect('equal')
    
    axes[1].arrow(0, 0, v1[0], v1[1], head_width=0.2, head_length=0.2,
                 fc='blue', ec='blue', linewidth=3, label='v1')
    axes[1].arrow(0, 0, v2[0], v2[1], head_width=0.2, head_length=0.2,
                 fc='red', ec='red', linewidth=2, alpha=0.7, label='v2')
    axes[1].arrow(0, 0, proj_v2_on_v1[0], proj_v2_on_v1[1], 
                 head_width=0.15, head_length=0.15,
                 fc='green', ec='green', linewidth=2, linestyle='--',
                 label=f'Projection of v2 on v1')
    axes[1].plot([v2[0], proj_v2_on_v1[0]], [v2[1], proj_v2_on_v1[1]],
                'k--', linewidth=1, alpha=0.5)
    
    axes[1].set_xlim(-1, max(4, max_val + 1))
    axes[1].set_ylim(-1, max(4, max_val + 1))
    axes[1].set_xlabel('X', fontsize=12)
    axes[1].set_ylabel('Y', fontsize=12)
    axes[1].set_title('Vector Projection', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linewidth=0.5)
    axes[1].axvline(x=0, color='k', linewidth=0.5)
    axes[1].legend(fontsize=10)
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    print(f"Dot product: v1 · v2 = {dot_product:.2f}")
    print(f"Angle: {angle_deg:.2f} degrees")
    print(f"Projection length: {np.linalg.norm(proj_v2_on_v1):.2f}")
    return fig


def visualize_cross_product_3d(v1: np.ndarray, v2: np.ndarray, save_path: Optional[str] = None):
    """Visualize cross product in 3D"""
    if len(v1) != 3 or len(v2) != 3:
        raise ValueError("Vectors must be 3D for cross product visualization")
    if Axes3D is None:
        raise ImportError("3D plotting requires mpl_toolkits.mplot3d")
    
    cross_product = np.cross(v1, v2)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='blue', arrow_length_ratio=0.2, 
              linewidth=3, label='v1')
    ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='red', arrow_length_ratio=0.2, 
              linewidth=3, label='v2')
    ax.quiver(0, 0, 0, cross_product[0], cross_product[1], cross_product[2], 
              color='green', arrow_length_ratio=0.2, linewidth=3, label='v1 × v2')
    
    max_val = max(np.abs(v1).max(), np.abs(v2).max(), np.abs(cross_product).max())
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('Cross Product in 3D', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim([-max_val-1, max_val+1])
    ax.set_ylim([-max_val-1, max_val+1])
    ax.set_zlim([-max_val-1, max_val+1])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def visualize_vector_magnitude(v: np.ndarray, save_path: Optional[str] = None):
    """Visualize vector magnitude"""
    magnitude = np.linalg.norm(v)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.arrow(0, 0, v[0], v[1], head_width=0.2, head_length=0.2,
             fc='blue', ec='blue', linewidth=3, label=f'v = {v}')
    ax.plot([0, v[0], v[0]], [0, 0, v[1]], 'r--', linewidth=1.5, alpha=0.7)
    ax.plot([v[0], v[0]], [0, v[1]], 'r-', linewidth=2, label=f'Magnitude = {magnitude:.2f}')
    
    ax.text(v[0]/2, -0.3, f'{v[0]}', fontsize=11, ha='center', fontweight='bold')
    ax.text(v[0] + 0.2, v[1]/2, f'{v[1]}', fontsize=11, va='center', fontweight='bold')
    ax.text(v[0]/2 + 0.2, v[1]/2 + 0.2, 
           f'||v|| = √({v[0]}² + {v[1]}²) = {magnitude:.2f}', 
           fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax.set_xlim(-0.5, v[0] + 1)
    ax.set_ylim(-0.5, v[1] + 1)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Vector Magnitude (Norm)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def visualize_scalar_multiplication(v: np.ndarray, scalars: List[float], 
                                   save_path: Optional[str] = None):
    """Visualize scalar multiplication"""
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for scalar, color in zip(scalars, colors):
        scaled_v = scalar * v
        ax.arrow(0, 0, scaled_v[0], scaled_v[1], head_width=0.15, head_length=0.15,
                fc=color, ec=color, linewidth=2, 
                label=f'{scalar} × v = {scaled_v}')
    
    max_val = max([abs(s * v).max() for s in scalars])
    ax.set_xlim(-max_val - 1, max_val + 1)
    ax.set_ylim(-max_val - 1, max_val + 1)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Scalar Multiplication', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ============================================================================
# MATRIX OPERATIONS
# ============================================================================

def matrix_operations_example():
    """Demonstrate basic matrix operations"""
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    M_sum = A + B
    print(f"Matrix addition:\n{A}\n+\n{B}\n=\n{M_sum}\n")
    
    M_product = np.dot(A, B)
    print(f"Matrix multiplication:\n{A}\n×\n{B}\n=\n{M_product}\n")
    
    A_transpose = A.T
    print(f"Matrix transpose:\n{A}\n^T\n=\n{A_transpose}\n")
    
    det_A = np.linalg.det(A)
    print(f"Determinant of A: {det_A}")
    
    if det_A != 0:
        A_inverse = np.linalg.inv(A)
        print(f"Inverse of A:\n{A_inverse}\n")
    
    return A, B, M_product, det_A


def visualize_matrix_multiplication(A: np.ndarray, B: np.ndarray, 
                                    save_path: Optional[str] = None):
    """Visualize matrix multiplication"""
    C = np.dot(A, B)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    im1 = axes[0].imshow(A, cmap='Blues', aspect='auto', vmin=0, vmax=10)
    axes[0].set_title('Matrix A', fontsize=12, fontweight='bold')
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            axes[0].text(j, i, A[i, j], ha='center', va='center', 
                        fontsize=14, fontweight='bold', color='white')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(B, cmap='Greens', aspect='auto', vmin=0, vmax=10)
    axes[1].set_title('Matrix B', fontsize=12, fontweight='bold')
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            axes[1].text(j, i, B[i, j], ha='center', va='center', 
                        fontsize=14, fontweight='bold', color='white')
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(C, cmap='Reds', aspect='auto', vmin=0, vmax=50)
    axes[2].set_title('A × B', fontsize=12, fontweight='bold')
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            axes[2].text(j, i, C[i, j], ha='center', va='center', 
                        fontsize=14, fontweight='bold', color='white')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def visualize_matrix_transpose(A: np.ndarray, save_path: Optional[str] = None):
    """Visualize matrix transpose"""
    A_transpose = A.T
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(A, cmap='Blues', aspect='auto')
    axes[0].set_title(f'Matrix A ({A.shape[0]}×{A.shape[1]})', fontsize=12, fontweight='bold')
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            axes[0].text(j, i, A[i, j], ha='center', va='center', 
                        fontsize=12, fontweight='bold', color='white')
    
    axes[1].imshow(A_transpose, cmap='Greens', aspect='auto')
    axes[1].set_title(f'A^T ({A_transpose.shape[0]}×{A_transpose.shape[1]})', 
                     fontsize=12, fontweight='bold')
    for i in range(A_transpose.shape[0]):
        for j in range(A_transpose.shape[1]):
            axes[1].text(j, i, A_transpose[i, j], ha='center', va='center', 
                        fontsize=12, fontweight='bold', color='white')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def visualize_determinant_geometric(A: np.ndarray, save_path: Optional[str] = None):
    """Visualize geometric interpretation of determinant"""
    unit_square = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
    transformed = A @ unit_square
    det_A = np.linalg.det(A)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].plot(unit_square[0], unit_square[1], 'b-', linewidth=2, marker='o', markersize=6)
    axes[0].fill(unit_square[0], unit_square[1], alpha=0.3, color='blue')
    axes[0].set_title('Original Unit Square\nArea = 1', fontsize=12, fontweight='bold')
    axes[0].set_xlim(-0.5, 3)
    axes[0].set_ylim(-0.5, 3)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    
    axes[1].plot(transformed[0], transformed[1], 'r-', linewidth=2, marker='o', markersize=6)
    axes[1].fill(transformed[0], transformed[1], alpha=0.3, color='red')
    axes[1].set_title(f'After Transformation A\nDet = {det_A:.2f}, Area = {abs(det_A):.2f}', 
                     fontsize=12, fontweight='bold')
    axes[1].set_xlim(-0.5, 3)
    axes[1].set_ylim(-0.5, 3)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ============================================================================
# EIGENVALUES AND EIGENVECTORS
# ============================================================================

def eigenvalues_eigenvectors_example():
    """Demonstrate eigenvalues and eigenvectors"""
    A = np.array([[4, 2], [2, 3]])
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print(f"Matrix A:\n{A}\n")
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}\n")
    
    for i, (eigenvalue, eigenvector) in enumerate(zip(eigenvalues, eigenvectors.T)):
        result = np.dot(A, eigenvector)
        expected = eigenvalue * eigenvector
        print(f"Verification {i+1}: Av = {result}, λv = {expected}")
        print(f"Match: {np.allclose(result, expected)}\n")
    
    return eigenvalues, eigenvectors


def visualize_eigenvectors(A: np.ndarray, save_path: Optional[str] = None):
    """Visualize eigenvectors and their transformations"""
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for i in range(len(eigenvalues)):
        vec = eigenvectors[:, i]
        eigenval = eigenvalues[i]
        
        ax.arrow(0, 0, vec[0], vec[1], head_width=0.15, head_length=0.15,
                fc='blue', ec='blue', linewidth=2, label=f'Eigenvector {i+1}')
        
        transformed = np.dot(A, vec)
        ax.arrow(0, 0, transformed[0], transformed[1], 
                head_width=0.15, head_length=0.15,
                fc='red', ec='red', linewidth=2, linestyle='--',
                label=f'A×v{i+1} (λ={eigenval:.2f})')
    
    max_val = max(abs(eigenvectors).max(), abs(eigenvalues * eigenvectors).max())
    ax.set_xlim(-max_val - 1, max_val + 1)
    ax.set_ylim(-max_val - 1, max_val + 1)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Eigenvectors and Their Transformations', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend(fontsize=9)
    ax.set_aspect('equal')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def visualize_eigenvalue_decomposition(A: np.ndarray, save_path: Optional[str] = None):
    """Visualize eigenvalue decomposition (circle to ellipse)"""
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    theta = np.linspace(0, 2*np.pi, 100)
    circle_vectors = np.array([np.cos(theta), np.sin(theta)])
    transformed_circle = A @ circle_vectors
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].plot(circle_vectors[0], circle_vectors[1], 'b-', linewidth=2, label='Unit Circle')
    axes[0].set_title('Original Unit Circle', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('X', fontsize=11)
    axes[0].set_ylabel('Y', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    axes[0].legend(fontsize=9)
    
    axes[1].plot(transformed_circle[0], transformed_circle[1], 'r-', linewidth=2, 
                label='Transformed Ellipse')
    
    for i, (eigenval, eigenvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        scaled = eigenval * eigenvec
        axes[1].arrow(0, 0, scaled[0], scaled[1], head_width=0.2, head_length=0.2,
                     fc=['blue', 'green'][i], ec=['blue', 'green'][i], linewidth=2,
                     label=f'Eigenvector {i+1} (λ={eigenval:.2f})')
    
    axes[1].set_title('After Transformation by A', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('X', fontsize=11)
    axes[1].set_ylabel('Y', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')
    axes[1].legend(fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ============================================================================
# LINEAR TRANSFORMATIONS
# ============================================================================

def visualize_linear_transformations(save_path: Optional[str] = None):
    """Visualize common linear transformations"""
    unit_square = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
    
    transformations = {
        'Identity': np.array([[1, 0], [0, 1]]),
        'Rotation 90°': np.array([[0, -1], [1, 0]]),
        'Reflection (y=x)': np.array([[0, 1], [1, 0]]),
        'Scaling': np.array([[2, 0], [0, 0.5]]),
        'Shear': np.array([[1, 1], [0, 1]])
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    axes[0].plot(unit_square[0], unit_square[1], 'b-', linewidth=2, marker='o', markersize=4)
    axes[0].fill(unit_square[0], unit_square[1], alpha=0.3, color='blue')
    axes[0].set_title('Original Unit Square', fontsize=11, fontweight='bold')
    axes[0].set_xlim(-2, 3)
    axes[0].set_ylim(-2, 3)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    
    for idx, (name, T) in enumerate(transformations.items(), 1):
        transformed = T @ unit_square
        axes[idx].plot(transformed[0], transformed[1], 'r-', linewidth=2, marker='o', markersize=4)
        axes[idx].fill(transformed[0], transformed[1], alpha=0.3, color='red')
        axes[idx].set_title(f'{name}\nDet = {np.linalg.det(T):.2f}', fontsize=11, fontweight='bold')
        axes[idx].set_xlim(-2, 3)
        axes[idx].set_ylim(-2, 3)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_aspect('equal')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def visualize_matrix_rank(A: np.ndarray, save_path: Optional[str] = None):
    """Visualize matrix rank"""
    rank = np.linalg.matrix_rank(A)
    col1 = A[:, 0]
    col2 = A[:, 1] if A.shape[1] > 1 else None
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.arrow(0, 0, col1[0], col1[1], head_width=0.2, head_length=0.2,
            fc='blue', ec='blue', linewidth=2, label='Column 1')
    
    if col2 is not None:
        ax.arrow(0, 0, col2[0], col2[1], head_width=0.2, head_length=0.2,
                fc='red', ec='red', linewidth=2, label='Column 2')
        is_dependent = np.allclose(np.cross(col1, col2), 0) if len(col1) == 2 else False
        title = f'Rank = {rank}\nColumns are {"dependent" if is_dependent else "independent"}'
    else:
        title = f'Rank = {rank}'
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    max_val = max(abs(col1).max(), abs(col2).max() if col2 is not None else 0)
    ax.set_xlim(-max_val - 1, max_val + 1)
    ax.set_ylim(-max_val - 1, max_val + 1)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_aspect('equal')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ============================================================================
# ADDITIONAL VISUALIZATIONS
# ============================================================================

def visualize_vector_space(vectors: List[np.ndarray], labels: List[str] = None, 
                          save_path: Optional[str] = None):
    """Visualize 2D vector space"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    
    for i, vec in enumerate(vectors):
        color = colors[i % len(colors)]
        label = labels[i] if labels and i < len(labels) else f'Vector {i+1}'
        ax.arrow(0, 0, vec[0], vec[1], head_width=0.2, head_length=0.2,
                fc=color, ec=color, linewidth=2, label=label)
        ax.text(vec[0]*1.1, vec[1]*1.1, label, fontsize=11, fontweight='bold')
    
    ax.arrow(0, 0, e1[0], e1[1], head_width=0.15, head_length=0.15,
            fc='black', ec='black', linewidth=2, linestyle='--', alpha=0.5)
    ax.arrow(0, 0, e2[0], e2[1], head_width=0.15, head_length=0.15,
            fc='black', ec='black', linewidth=2, linestyle='--', alpha=0.5)
    ax.text(1.1, 0, 'e1', fontsize=10, style='italic')
    ax.text(0, 1.1, 'e2', fontsize=10, style='italic')
    
    max_val = max([abs(v).max() for v in vectors]) if vectors else 4
    ax.set_xlim(-max_val - 1, max_val + 1)
    ax.set_ylim(-max_val - 1, max_val + 1)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('2D Vector Space', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def visualize_orthogonal_vectors(v1: np.ndarray, v2: np.ndarray, 
                                 save_path: Optional[str] = None):
    """Visualize orthogonal vectors"""
    dot_product = np.dot(v1, v2)
    is_orthogonal = abs(dot_product) < 1e-10
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.arrow(0, 0, v1[0], v1[1], head_width=0.2, head_length=0.2,
            fc='blue', ec='blue', linewidth=3, label=f'v1 = {v1}')
    ax.arrow(0, 0, v2[0], v2[1], head_width=0.2, head_length=0.2,
            fc='red', ec='red', linewidth=3, label=f'v2 = {v2}')
    
    if is_orthogonal:
        ax.plot([0.3, 0.3, 0], [0, 0.2, 0.2], 'k-', linewidth=1.5)
    
    max_val = max(abs(v1).max(), abs(v2).max())
    ax.set_xlim(-max_val - 1, max_val + 1)
    ax.set_ylim(-max_val - 1, max_val + 1)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(f'Orthogonal Vectors\nv1 · v2 = {dot_product:.2f} (should be 0)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=== Vector Operations ===")
    vector_operations_example()
    
    print("\n=== Matrix Operations ===")
    matrix_operations_example()
    
    print("\n=== Eigenvalues and Eigenvectors ===")
    eigenvalues_eigenvectors_example()
    
    print("\n=== Generating Visualizations ===")
    print("Run individual visualization functions to see graphs!")
