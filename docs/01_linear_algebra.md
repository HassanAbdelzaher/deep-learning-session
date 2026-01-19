# Linear Algebra for Deep Learning

## Table of Contents
1. [Introduction](#introduction)
2. [Vectors](#vectors)
3. [Matrices](#matrices)
4. [Eigenvalues and Eigenvectors](#eigenvalues-and-eigenvectors)
5. [Visualizations](#visualizations)

## Introduction

Linear algebra is the foundation of deep learning. Neural networks are essentially a series of matrix multiplications and transformations. Understanding vectors, matrices, and their operations is crucial for understanding how neural networks work.

## Vectors

### What is a Vector?

A vector is an ordered collection of numbers. In deep learning, vectors represent:
- Input features
- Weights in neural networks
- Gradients during optimization
- Activations in layers

### Vector Operations

#### Vector Addition

```python
import numpy as np
import matplotlib.pyplot as plt

# Create vectors
v1 = np.array([1, 2])
v2 = np.array([3, 1])

# Vector addition
v_sum = v1 + v2
print(f"v1 + v2 = {v_sum}")  # [4, 3]

# Visualize
fig, ax = plt.subplots(figsize=(8, 8))
ax.arrow(0, 0, v1[0], v1[1], head_width=0.1, head_length=0.1, 
         fc='blue', ec='blue', label='v1', linewidth=2)
ax.arrow(0, 0, v2[0], v2[1], head_width=0.1, head_length=0.1, 
         fc='green', ec='green', label='v2', linewidth=2)
ax.arrow(0, 0, v_sum[0], v_sum[1], head_width=0.1, head_length=0.1, 
         fc='red', ec='red', label='v1 + v2', linewidth=2)
ax.arrow(v1[0], v1[1], v2[0], v2[1], head_width=0.1, head_length=0.1, 
         fc='orange', ec='orange', linestyle='--', alpha=0.5, linewidth=1)

ax.set_xlim(-1, 5)
ax.set_ylim(-1, 5)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Vector Addition', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.legend(fontsize=10)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('docs/images/vector_addition.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Graph Explanation**: The red vector shows the result of adding v1 (blue) and v2 (green). The orange dashed line shows the parallelogram rule for vector addition.

#### Dot Product (Scalar Product)

The dot product measures how much two vectors point in the same direction.

```python
# Dot product
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
dot_product = np.dot(v1, v2)
print(f"v1 · v2 = {dot_product}")  # 1*4 + 2*5 + 3*6 = 32

# Geometric interpretation: ||v1|| * ||v2|| * cos(θ)
magnitude_v1 = np.linalg.norm(v1)
magnitude_v2 = np.linalg.norm(v2)
cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
angle_rad = np.arccos(cos_theta)
angle_deg = np.degrees(angle_rad)
print(f"Angle between vectors: {angle_deg:.2f} degrees")
```

#### Cross Product (3D only)

```python
# Cross product (perpendicular to both vectors)
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
cross_product = np.cross(v1, v2)
print(f"v1 × v2 = {cross_product}")  # [-3, 6, -3]
```

## Matrices

### What is a Matrix?

A matrix is a 2D array of numbers. In deep learning:
- Weight matrices connect layers
- Input data is often represented as matrices
- Transformations are matrix multiplications

### Matrix Operations

#### Matrix Multiplication

Matrix multiplication is the core operation in neural networks. Each layer performs: `output = input × weights + bias`

```python
# Matrix multiplication
A = np.array([[1, 2], [3, 4]])  # 2x2
B = np.array([[5, 6], [7, 8]])  # 2x2
C = np.dot(A, B)
print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)
print("\nA × B:")
print(C)

# Visualize matrix multiplication
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Matrix A
im1 = axes[0].imshow(A, cmap='Blues', aspect='auto', vmin=0, vmax=10)
axes[0].set_title('Matrix A', fontsize=12, fontweight='bold')
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(['1', '2'])
axes[0].set_yticklabels(['1', '2'])
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, A[i, j], ha='center', va='center', 
                     fontsize=14, fontweight='bold', color='white')
plt.colorbar(im1, ax=axes[0])

# Matrix B
im2 = axes[1].imshow(B, cmap='Greens', aspect='auto', vmin=0, vmax=10)
axes[1].set_title('Matrix B', fontsize=12, fontweight='bold')
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['1', '2'])
axes[1].set_yticklabels(['1', '2'])
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, B[i, j], ha='center', va='center', 
                     fontsize=14, fontweight='bold', color='white')
plt.colorbar(im2, ax=axes[1])

# Result C
im3 = axes[2].imshow(C, cmap='Reds', aspect='auto', vmin=0, vmax=50)
axes[2].set_title('A × B', fontsize=12, fontweight='bold')
axes[2].set_xticks([0, 1])
axes[2].set_yticks([0, 1])
axes[2].set_xticklabels(['1', '2'])
axes[2].set_yticklabels(['1', '2'])
for i in range(2):
    for j in range(2):
        axes[2].text(j, i, C[i, j], ha='center', va='center', 
                     fontsize=14, fontweight='bold', color='white')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig('docs/images/matrix_multiplication.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Key Insight**: In matrix multiplication `C = A × B`, element `C[i,j]` is the dot product of row `i` of A and column `j` of B.

#### Matrix Transpose

```python
A = np.array([[1, 2, 3], [4, 5, 6]])
A_transpose = A.T
print("Original Matrix A:")
print(A)
print("\nTranspose A^T:")
print(A_transpose)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(A, cmap='Blues', aspect='auto')
axes[0].set_title('Matrix A (2×3)', fontsize=12, fontweight='bold')
for i in range(2):
    for j in range(3):
        axes[0].text(j, i, A[i, j], ha='center', va='center', 
                     fontsize=12, fontweight='bold', color='white')
axes[0].set_xticks(range(3))
axes[0].set_yticks(range(2))

axes[1].imshow(A_transpose, cmap='Greens', aspect='auto')
axes[1].set_title('A^T (3×2)', fontsize=12, fontweight='bold')
for i in range(3):
    for j in range(2):
        axes[1].text(j, i, A_transpose[i, j], ha='center', va='center', 
                     fontsize=12, fontweight='bold', color='white')
axes[1].set_xticks(range(2))
axes[1].set_yticks(range(3))

plt.tight_layout()
plt.savefig('docs/images/matrix_transpose.png', dpi=150, bbox_inches='tight')
plt.show()
```

#### Determinant and Inverse

```python
A = np.array([[4, 2], [2, 3]])

# Determinant
det_A = np.linalg.det(A)
print(f"Determinant of A: {det_A:.2f}")

# Inverse (only if determinant ≠ 0)
if det_A != 0:
    A_inverse = np.linalg.inv(A)
    print("\nInverse of A:")
    print(A_inverse)
    
    # Verify: A × A^-1 = I (identity matrix)
    identity = np.dot(A, A_inverse)
    print("\nA × A^-1 (should be identity):")
    print(identity)
```

## Eigenvalues and Eigenvectors

Eigenvalues and eigenvectors are crucial in:
- Principal Component Analysis (PCA)
- Understanding neural network dynamics
- Matrix decomposition

### Definition

For a matrix A, if `Av = λv` where:
- `v` is an eigenvector (non-zero vector)
- `λ` is an eigenvalue (scalar)

Then `v` is an eigenvector with eigenvalue `λ`.

```python
# Find eigenvalues and eigenvectors
A = np.array([[4, 2], [2, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Matrix A:")
print(A)
print(f"\nEigenvalues: {eigenvalues}")
print(f"\nEigenvectors (columns):")
print(eigenvectors)

# Visualize eigenvectors
fig, ax = plt.subplots(figsize=(10, 10))

# Original vectors
for i in range(2):
    vec = eigenvectors[:, i]
    eigenval = eigenvalues[i]
    
    # Original vector
    ax.arrow(0, 0, vec[0], vec[1], head_width=0.15, head_length=0.15,
             fc='blue', ec='blue', linewidth=2, label=f'Eigenvector {i+1}')
    
    # Transformed vector (should be parallel)
    transformed = np.dot(A, vec)
    ax.arrow(0, 0, transformed[0], transformed[1], 
             head_width=0.15, head_length=0.15,
             fc='red', ec='red', linewidth=2, linestyle='--',
             label=f'A×v{i+1} (λ={eigenval:.2f})')
    
    # Verify: transformed should be eigenvalue * original
    scaled = eigenval * vec
    ax.arrow(0, 0, scaled[0], scaled[1], 
             head_width=0.1, head_length=0.1,
             fc='green', ec='green', linewidth=1.5, linestyle=':',
             alpha=0.7)

ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Eigenvectors and Their Transformations', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.legend(fontsize=9)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('docs/images/eigenvectors.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Key Insight**: When a matrix multiplies its eigenvector, the result is just a scaled version of the eigenvector (scaled by the eigenvalue).

## Visualizations

### Vector Space Visualization

```python
# Create a 2D vector space
fig, ax = plt.subplots(figsize=(10, 10))

# Basis vectors
e1 = np.array([1, 0])
e2 = np.array([0, 1])

# Some vectors
vectors = [
    np.array([2, 1]),
    np.array([-1, 2]),
    np.array([3, -1]),
    np.array([-2, -2])
]

colors = ['blue', 'green', 'red', 'orange']
labels = ['v1', 'v2', 'v3', 'v4']

for vec, color, label in zip(vectors, colors, labels):
    ax.arrow(0, 0, vec[0], vec[1], head_width=0.2, head_length=0.2,
             fc=color, ec=color, linewidth=2, label=label)
    ax.text(vec[0]*1.1, vec[1]*1.1, label, fontsize=11, fontweight='bold')

# Basis vectors
ax.arrow(0, 0, e1[0], e1[1], head_width=0.15, head_length=0.15,
         fc='black', ec='black', linewidth=2, linestyle='--', alpha=0.5)
ax.arrow(0, 0, e2[0], e2[1], head_width=0.15, head_length=0.15,
         fc='black', ec='black', linewidth=2, linestyle='--', alpha=0.5)
ax.text(1.1, 0, 'e1', fontsize=10, style='italic')
ax.text(0, 1.1, 'e2', fontsize=10, style='italic')

ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('2D Vector Space', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.legend(fontsize=10)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('docs/images/vector_space.png', dpi=150, bbox_inches='tight')
plt.show()
```

## Practice Exercises

1. **Vector Operations**: Given `v1 = [1, 3, 5]` and `v2 = [2, 4, 6]`, calculate:
   - `v1 + v2`
   - `v1 · v2` (dot product)
   - `v1 × v2` (cross product)

2. **Matrix Multiplication**: Given `A = [[1, 2], [3, 4]]` and `B = [[5, 6], [7, 8]]`, calculate `A × B`.

3. **Eigenvalues**: Find the eigenvalues and eigenvectors of `[[2, 1], [1, 2]]`.

## Summary

- **Vectors** are 1D arrays representing direction and magnitude
- **Matrices** are 2D arrays used for transformations
- **Matrix multiplication** is the core operation in neural networks
- **Eigenvalues/eigenvectors** help understand matrix transformations
- All neural network operations can be expressed as matrix operations

## Next Steps

- Study [Calculus for Deep Learning](02_calculus.md)
- Learn about [Neural Networks](../docs/04_neural_networks.md)
