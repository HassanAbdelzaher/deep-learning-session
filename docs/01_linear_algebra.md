# Linear Algebra for Deep Learning

## Table of Contents
1. [Introduction](#introduction)
2. [Vectors](#vectors)
   - Vector Addition
   - Dot Product with Visualizations
   - Cross Product (3D)
   - Vector Magnitude
   - Scalar Multiplication
3. [Matrices](#matrices)
   - Matrix Multiplication
   - Matrix Transpose
   - Determinant and Inverse (with Geometric Interpretation)
4. [Eigenvalues and Eigenvectors](#eigenvalues-and-eigenvectors)
   - Definition and Visualization
   - Eigenvalue Decomposition
5. [Linear Transformations](#linear-transformations)
   - Common Transformations
   - Matrix Rank
6. [Visualizations](#visualizations)
   - Vector Space
   - Linear Combinations
   - Orthogonal Vectors
   - Matrix Multiplication Step-by-Step
7. [Linear Algebra in Image Processing](#linear-algebra-in-image-processing) ⭐ NEW
   - Image Representation as Matrices
   - Image Transformations
   - Image Filtering (Convolution)
   - Image Compression (SVD)
   - Edge Detection
   - Image Enhancement
   - Color Image Processing

## Introduction

Linear algebra is the foundation of deep learning. Neural networks are essentially a series of matrix multiplications and transformations. Understanding vectors, matrices, and their operations is crucial for understanding how neural networks work.

**Mermaid Diagram: Linear Algebra in Neural Networks**

```mermaid
flowchart TD
    A["Input Data<br/>(Vector/Matrix)"] --> B["Layer 1:<br/>W₁ × x + b₁"]
    B --> C["Activation Function<br/>σ(z)"]
    C --> D["Layer 2:<br/>W₂ × a₁ + b₂"]
    D --> E["Output<br/>(Predictions)"]
    
    F["Key Operations:"] --> G["Matrix Multiplication<br/>W × x"]
    F --> H["Vector Addition<br/>+ b"]
    F --> I["Element-wise Operations<br/>σ(z)"]
    
    J["Why Linear Algebra?"] --> K["Efficient Computation<br/>(Vectorized)"]
    J --> L["Parallel Processing<br/>(GPU-friendly)"]
    J --> M["Mathematical Foundation<br/>(Gradients, Optimization)"]
    
    style A fill:#e1f5ff
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style D fill:#fff3e0
    style E fill:#e8f5e9
    style F fill:#e0f2f1
    style G fill:#e0f2f1
    style H fill:#e0f2f1
    style I fill:#e0f2f1
    style J fill:#fff9c4
    style K fill:#fff9c4
    style L fill:#fff9c4
    style M fill:#fff9c4
```

## Vectors

### What is a Vector?

A vector is an ordered collection of numbers. In deep learning, vectors represent:
- Input features
- Weights in neural networks
- Gradients during optimization
- Activations in layers

**Mermaid Diagram: Vector Representation**

```mermaid
graph LR
    A["Vector v = [v₁, v₂, ..., vₙ]"] --> B["Geometric:<br/>Arrow in space"]
    A --> C["Algebraic:<br/>Ordered tuple"]
    A --> D["Computational:<br/>Array/List"]
    
    E["Properties:"] --> F["Magnitude: ||v|| = √(v₁² + v₂² + ...)"]
    E --> G["Direction: Angle from axes"]
    E --> H["Dimension: Number of components"]
    
    style A fill:#e1f5ff
    style B fill:#e8f5e9
    style C fill:#fff3e0
    style D fill:#f3e5f5
    style E fill:#e0f2f1
    style F fill:#e0f2f1
    style G fill:#e0f2f1
    style H fill:#e0f2f1
```

### Vector Operations

#### Vector Addition

**Mermaid Diagram: Vector Addition Process**

```mermaid
flowchart LR
    A["Vector v₁<br/>[x₁, y₁]"] --> C["Add Components"]
    B["Vector v₂<br/>[x₂, y₂]"] --> C
    C --> D["Result v₁ + v₂<br/>[x₁+x₂, y₁+y₂]"]
    style A fill:#e1f5ff
    style B fill:#e8f5e9
    style C fill:#fff3e0
    style D fill:#ffebee
```

**Geometric Visualization:**

```mermaid
graph TB
    subgraph "Vector Addition: v₁ + v₂"
        O["Origin (0,0)"] -->|"v₁"| P1["Point (x₁, y₁)"]
        O -->|"v₂"| P2["Point (x₂, y₂)"]
        O -->|"v₁ + v₂"| P3["Point (x₁+x₂, y₁+y₂)"]
        P1 -->|"Parallelogram Rule"| P3
    end
    style O fill:#f9f9f9
    style P1 fill:#e1f5ff
    style P2 fill:#e8f5e9
    style P3 fill:#ffebee
```

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

**Mermaid Diagram: Dot Product Calculation**

```mermaid
flowchart TD
    A["Vector v₁ = [a₁, a₂, ..., aₙ]"] --> C["Multiply<br/>Component-wise"]
    B["Vector v₂ = [b₁, b₂, ..., bₙ]"] --> C
    C --> D["Sum Products<br/>a₁b₁ + a₂b₂ + ... + aₙbₙ"]
    D --> E["Result: Scalar Value"]
    
    F["Geometric Formula"] --> G["v₁ · v₂ = ||v₁|| × ||v₂|| × cos(θ)"]
    
    style A fill:#e1f5ff
    style B fill:#e8f5e9
    style C fill:#fff3e0
    style D fill:#f3e5f5
    style E fill:#ffebee
    style F fill:#e0f2f1
    style G fill:#e0f2f1
```

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

**Visualization of Dot Product with Angle:**

```python
# Visualize dot product and angle between vectors
v1 = np.array([3, 2])
v2 = np.array([2, 3])

dot_product = np.dot(v1, v2)
magnitude_v1 = np.linalg.norm(v1)
magnitude_v2 = np.linalg.norm(v2)
angle_rad = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))
angle_deg = np.degrees(angle_rad)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Vector visualization with angle
axes[0].arrow(0, 0, v1[0], v1[1], head_width=0.2, head_length=0.2,
             fc='blue', ec='blue', linewidth=3, label=f'v1 = {v1}')
axes[0].arrow(0, 0, v2[0], v2[1], head_width=0.2, head_length=0.2,
             fc='red', ec='red', linewidth=3, label=f'v2 = {v2}')

# Draw angle arc
theta = np.linspace(0, angle_rad, 100)
r = 0.8
axes[0].plot(r * np.cos(theta), r * np.sin(theta), 'g-', linewidth=2)
axes[0].text(0.5, 0.3, f'θ = {angle_deg:.1f}°', fontsize=12, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

axes[0].set_xlim(-1, 4)
axes[0].set_ylim(-1, 4)
axes[0].set_xlabel('X', fontsize=12)
axes[0].set_ylabel('Y', fontsize=12)
axes[0].set_title('Dot Product: Angle Between Vectors', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='k', linewidth=0.5)
axes[0].axvline(x=0, color='k', linewidth=0.5)
axes[0].legend(fontsize=10)
axes[0].set_aspect('equal')

# Projection visualization
proj_v2_on_v1 = (dot_product / (magnitude_v1**2)) * v1
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

axes[1].set_xlim(-1, 4)
axes[1].set_ylim(-1, 4)
axes[1].set_xlabel('X', fontsize=12)
axes[1].set_ylabel('Y', fontsize=12)
axes[1].set_title('Vector Projection', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='k', linewidth=0.5)
axes[1].axvline(x=0, color='k', linewidth=0.5)
axes[1].legend(fontsize=10)
axes[1].set_aspect('equal')

plt.tight_layout()
plt.savefig('docs/images/dot_product_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Dot product: v1 · v2 = {dot_product:.2f}")
print(f"Angle: {angle_deg:.2f} degrees")
print(f"Projection length: {np.linalg.norm(proj_v2_on_v1):.2f}")
```

#### Cross Product (3D only)

The cross product produces a vector perpendicular to both input vectors.

**Mermaid Diagram: Cross Product**

```mermaid
flowchart LR
    A["v₁ = [x₁, y₁, z₁]"] --> C["Cross Product<br/>v₁ × v₂"]
    B["v₂ = [x₂, y₂, z₂]"] --> C
    C --> D["Result: Perpendicular Vector<br/>[y₁z₂ - z₁y₂,<br/>z₁x₂ - x₁z₂,<br/>x₁y₂ - y₁x₂]"]
    
    E["Right-Hand Rule"] --> F["Thumb: v₁<br/>Index: v₂<br/>Middle: Result"]
    
    style A fill:#e1f5ff
    style B fill:#e8f5e9
    style C fill:#fff3e0
    style D fill:#ffebee
    style E fill:#f3e5f5
    style F fill:#f3e5f5
```

```python
# Cross product (perpendicular to both vectors)
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
cross_product = np.cross(v1, v2)
print(f"v1 × v2 = {cross_product}")  # [-3, 6, -3]
```

**3D Visualization of Cross Product:**

```python
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plots

# Create 3D vectors
v1 = np.array([2, 0, 0])
v2 = np.array([0, 2, 0])
cross_product = np.cross(v1, v2)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Draw vectors
ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='blue', arrow_length_ratio=0.2, linewidth=3, label='v1')
ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='red', arrow_length_ratio=0.2, linewidth=3, label='v2')
ax.quiver(0, 0, 0, cross_product[0], cross_product[1], cross_product[2], 
         color='green', arrow_length_ratio=0.2, linewidth=3, label='v1 × v2')

# Draw plane spanned by v1 and v2
xx, yy = np.meshgrid(np.linspace(-0.5, 2.5, 10), np.linspace(-0.5, 2.5, 10))
zz = np.zeros_like(xx)
ax.plot_surface(xx, yy, zz, alpha=0.2, color='yellow')

ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_zlabel('Z', fontsize=12)
ax.set_title('Cross Product in 3D', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim([-1, 3])
ax.set_ylim([-1, 3])
ax.set_zlim([-1, 3])
plt.tight_layout()
plt.savefig('docs/images/cross_product_3d.png', dpi=150, bbox_inches='tight')
plt.show()
```

#### Vector Magnitude (Norm)

```python
# Vector magnitude visualization
v = np.array([3, 4])
magnitude = np.linalg.norm(v)

fig, ax = plt.subplots(figsize=(8, 8))
ax.arrow(0, 0, v[0], v[1], head_width=0.2, head_length=0.2,
         fc='blue', ec='blue', linewidth=3, label=f'v = {v}')

# Draw right triangle
ax.plot([0, v[0], v[0]], [0, 0, v[1]], 'r--', linewidth=1.5, alpha=0.7)
ax.plot([v[0], v[0]], [0, v[1]], 'r-', linewidth=2, label=f'Magnitude = {magnitude:.2f}')

# Annotations
ax.text(v[0]/2, -0.3, f'{v[0]}', fontsize=11, ha='center', fontweight='bold')
ax.text(v[0] + 0.2, v[1]/2, f'{v[1]}', fontsize=11, va='center', fontweight='bold')
ax.text(v[0]/2 + 0.2, v[1]/2 + 0.2, f'||v|| = √({v[0]}² + {v[1]}²) = {magnitude:.2f}', 
       fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

ax.set_xlim(-0.5, 4)
ax.set_ylim(-0.5, 5)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Vector Magnitude (Norm)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.legend(fontsize=10)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('docs/images/vector_magnitude.png', dpi=150, bbox_inches='tight')
plt.show()
```

#### Scalar Multiplication

```python
# Scalar multiplication visualization
v = np.array([2, 1])
scalars = [0.5, 1, 2, -1]

fig, ax = plt.subplots(figsize=(10, 10))
colors = ['red', 'blue', 'green', 'orange']

for scalar, color in zip(scalars, colors):
    scaled_v = scalar * v
    ax.arrow(0, 0, scaled_v[0], scaled_v[1], head_width=0.15, head_length=0.15,
            fc=color, ec=color, linewidth=2, 
            label=f'{scalar} × v = {scaled_v}')

ax.set_xlim(-3, 5)
ax.set_ylim(-3, 3)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Scalar Multiplication', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.legend(fontsize=10)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('docs/images/scalar_multiplication.png', dpi=150, bbox_inches='tight')
plt.show()
```

## Matrices

### What is a Matrix?

A matrix is a 2D array of numbers. In deep learning:
- Weight matrices connect layers
- Input data is often represented as matrices
- Transformations are matrix multiplications

**Mermaid Diagram: Matrix Structure**

```mermaid
graph TB
    A["Matrix A<br/>m rows × n columns"] --> B["A[i,j] = element at<br/>row i, column j"]
    
    C["Matrix Types:"] --> D["Square: m = n"]
    C --> E["Rectangular: m ≠ n"]
    C --> F["Identity: I[i,i] = 1,<br/>I[i,j] = 0 if i≠j"]
    C --> G["Diagonal: Non-zero only<br/>on main diagonal"]
    
    H["In Neural Networks:"] --> I["Weight Matrix W:<br/>Connects layers"]
    H --> J["Input Matrix X:<br/>Batch of samples"]
    H --> K["Output Matrix Y:<br/>Predictions"]
    
    style A fill:#e1f5ff
    style B fill:#fff3e0
    style C fill:#e0f2f1
    style D fill:#e0f2f1
    style E fill:#e0f2f1
    style F fill:#e0f2f1
    style G fill:#e0f2f1
    style H fill:#fff9c4
    style I fill:#fff9c4
    style J fill:#fff9c4
    style K fill:#fff9c4
```

### Matrix Operations

#### Matrix Multiplication

Matrix multiplication is the core operation in neural networks. Each layer performs: `output = input × weights + bias`

**Mermaid Diagram: Matrix Multiplication Process**

```mermaid
flowchart TD
    A["Matrix A<br/>m × n"] --> C["Matrix<br/>Multiplication"]
    B["Matrix B<br/>n × p"] --> C
    C --> D["Result Matrix C<br/>m × p"]
    
    E["For each element C[i,j]:"] --> F["C[i,j] = Row i of A · Column j of B"]
    F --> G["= Σ A[i,k] × B[k,j]<br/>for k = 1 to n"]
    
    style A fill:#e1f5ff
    style B fill:#e8f5e9
    style C fill:#fff3e0
    style D fill:#ffebee
    style E fill:#f3e5f5
    style F fill:#f3e5f5
    style G fill:#f3e5f5
```

**Step-by-Step Matrix Multiplication:**

```mermaid
graph TB
    subgraph "Matrix A (2×2)"
        A11["A[0,0]"] 
        A12["A[0,1]"]
        A21["A[1,0]"]
        A22["A[1,1]"]
    end
    
    subgraph "Matrix B (2×2)"
        B11["B[0,0]"]
        B12["B[0,1]"]
        B21["B[1,0]"]
        B22["B[1,1]"]
    end
    
    subgraph "Result C = A × B"
        C11["C[0,0] = A[0,0]×B[0,0] + A[0,1]×B[1,0]"]
        C12["C[0,1] = A[0,0]×B[0,1] + A[0,1]×B[1,1]"]
        C21["C[1,0] = A[1,0]×B[0,0] + A[1,1]×B[1,0]"]
        C22["C[1,1] = A[1,0]×B[0,1] + A[1,1]×B[1,1]"]
    end
    
    A11 --> C11
    A12 --> C11
    B11 --> C11
    B21 --> C11
    
    A11 --> C12
    A12 --> C12
    B12 --> C12
    B22 --> C12
    
    A21 --> C21
    A22 --> C21
    B11 --> C21
    B21 --> C21
    
    A21 --> C22
    A22 --> C22
    B12 --> C22
    B22 --> C22
    
    style A11 fill:#e1f5ff
    style A12 fill:#e1f5ff
    style A21 fill:#e1f5ff
    style A22 fill:#e1f5ff
    style B11 fill:#e8f5e9
    style B12 fill:#e8f5e9
    style B21 fill:#e8f5e9
    style B22 fill:#e8f5e9
    style C11 fill:#ffebee
    style C12 fill:#ffebee
    style C21 fill:#ffebee
    style C22 fill:#ffebee
```

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

**Mermaid Diagram: Matrix Transpose Operation**

```mermaid
flowchart LR
    A["Matrix A<br/>m × n<br/>A[i,j]"] --> B["Transpose Operation<br/>Swap rows and columns"]
    B --> C["Matrix Aᵀ<br/>n × m<br/>Aᵀ[j,i] = A[i,j]"]
    
    D["Example:"] --> E["A = [[1, 2, 3],<br/>     [4, 5, 6]]<br/>2×3"]
    E --> F["Aᵀ = [[1, 4],<br/>      [2, 5],<br/>      [3, 6]]<br/>3×2"]
    
    style A fill:#e1f5ff
    style B fill:#fff3e0
    style C fill:#e8f5e9
    style D fill:#f3e5f5
    style E fill:#f3e5f5
    style F fill:#f3e5f5
```

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

The determinant measures how much a matrix scales area/volume. The inverse undoes the transformation.

**Mermaid Diagram: Determinant and Inverse**

```mermaid
flowchart TD
    A["Matrix A"] --> B{"Calculate<br/>Determinant"}
    B -->|"det(A) ≠ 0"| C["Matrix is<br/>Invertible"]
    B -->|"det(A) = 0"| D["Matrix is<br/>Singular<br/>(No Inverse)"]
    
    C --> E["Calculate A⁻¹"]
    E --> F["Verify:<br/>A × A⁻¹ = I<br/>(Identity Matrix)"]
    
    G["Geometric Meaning:"] --> H["det(A) = Area/Volume<br/>scaling factor"]
    H --> I["det(A) > 0: Preserves orientation"]
    H --> J["det(A) < 0: Reverses orientation"]
    
    style A fill:#e1f5ff
    style B fill:#fff3e0
    style C fill:#e8f5e9
    style D fill:#ffebee
    style E fill:#f3e5f5
    style F fill:#f3e5f5
    style G fill:#e0f2f1
    style H fill:#e0f2f1
    style I fill:#e0f2f1
    style J fill:#e0f2f1
```

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

**Geometric Interpretation of Determinant:**

```python
# Visualize how determinant affects area
A1 = np.array([[2, 0], [0, 2]])  # Determinant = 4 (scales by 2x2)
A2 = np.array([[1, 1], [0, 1]])  # Determinant = 1 (shear, area preserved)
A3 = np.array([[0.5, 0], [0, 0.5]])  # Determinant = 0.25 (shrinks)

# Unit square
unit_square = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Original unit square
axes[0, 0].plot(unit_square[0], unit_square[1], 'b-', linewidth=2, marker='o', markersize=6)
axes[0, 0].fill(unit_square[0], unit_square[1], alpha=0.3, color='blue')
axes[0, 0].set_title('Original Unit Square\nArea = 1', fontsize=12, fontweight='bold')
axes[0, 0].set_xlim(-0.5, 3)
axes[0, 0].set_ylim(-0.5, 3)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_aspect('equal')

# Transformed by A1
transformed1 = A1 @ unit_square
det1 = np.linalg.det(A1)
axes[0, 1].plot(transformed1[0], transformed1[1], 'r-', linewidth=2, marker='o', markersize=6)
axes[0, 1].fill(transformed1[0], transformed1[1], alpha=0.3, color='red')
axes[0, 1].set_title(f'After A1 (Scaling)\nDet = {det1:.2f}, Area = {det1:.2f}', 
                   fontsize=12, fontweight='bold')
axes[0, 1].set_xlim(-0.5, 3)
axes[0, 1].set_ylim(-0.5, 3)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_aspect('equal')

# Transformed by A2
transformed2 = A2 @ unit_square
det2 = np.linalg.det(A2)
axes[1, 0].plot(transformed2[0], transformed2[1], 'g-', linewidth=2, marker='o', markersize=6)
axes[1, 0].fill(transformed2[0], transformed2[1], alpha=0.3, color='green')
axes[1, 0].set_title(f'After A2 (Shear)\nDet = {det2:.2f}, Area = {det2:.2f}', 
                   fontsize=12, fontweight='bold')
axes[1, 0].set_xlim(-0.5, 3)
axes[1, 0].set_ylim(-0.5, 3)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_aspect('equal')

# Transformed by A3
transformed3 = A3 @ unit_square
det3 = np.linalg.det(A3)
axes[1, 1].plot(transformed3[0], transformed3[1], 'orange', linewidth=2, marker='o', markersize=6)
axes[1, 1].fill(transformed3[0], transformed3[1], alpha=0.3, color='orange')
axes[1, 1].set_title(f'After A3 (Shrink)\nDet = {det3:.2f}, Area = {det3:.2f}', 
                   fontsize=12, fontweight='bold')
axes[1, 1].set_xlim(-0.5, 3)
axes[1, 1].set_ylim(-0.5, 3)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_aspect('equal')

plt.tight_layout()
plt.savefig('docs/images/determinant_geometric.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Matrix Inverse Visualization:**

```python
# Visualize matrix and its inverse
A = np.array([[2, 1], [1, 2]])
A_inv = np.linalg.inv(A)

# Test vector
v = np.array([1, 1])

# Transformations
v_transformed = A @ v
v_restored = A_inv @ v_transformed

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original
axes[0].arrow(0, 0, v[0], v[1], head_width=0.15, head_length=0.15,
            fc='blue', ec='blue', linewidth=3, label='Original v')
axes[0].set_title('Original Vector v', fontsize=12, fontweight='bold')
axes[0].set_xlim(-1, 4)
axes[0].set_ylim(-1, 4)
axes[0].grid(True, alpha=0.3)
axes[0].set_aspect('equal')
axes[0].legend(fontsize=9)

# After A
axes[1].arrow(0, 0, v_transformed[0], v_transformed[1], 
            head_width=0.15, head_length=0.15,
            fc='red', ec='red', linewidth=3, label='A × v')
axes[1].set_title('After Transformation A', fontsize=12, fontweight='bold')
axes[1].set_xlim(-1, 4)
axes[1].set_ylim(-1, 4)
axes[1].grid(True, alpha=0.3)
axes[1].set_aspect('equal')
axes[1].legend(fontsize=9)

# After A^-1 (restored)
axes[2].arrow(0, 0, v_restored[0], v_restored[1], 
            head_width=0.15, head_length=0.15,
            fc='green', ec='green', linewidth=3, label='A⁻¹ × (A × v)')
axes[2].arrow(0, 0, v[0], v[1], head_width=0.1, head_length=0.1,
            fc='blue', ec='blue', linewidth=2, linestyle='--', alpha=0.5, label='Original v')
axes[2].set_title('After Inverse A⁻¹ (Restored)', fontsize=12, fontweight='bold')
axes[2].set_xlim(-1, 4)
axes[2].set_ylim(-1, 4)
axes[2].grid(True, alpha=0.3)
axes[2].set_aspect('equal')
axes[2].legend(fontsize=9)

plt.tight_layout()
plt.savefig('docs/images/matrix_inverse.png', dpi=150, bbox_inches='tight')
plt.show()
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

**Mermaid Diagram: Eigenvalue and Eigenvector Relationship**

```mermaid
flowchart TD
    A["Matrix A"] --> B["Find Eigenvalues λ<br/>Solve: det(A - λI) = 0"]
    B --> C["For each eigenvalue λ"]
    C --> D["Find Eigenvector v<br/>Solve: (A - λI)v = 0"]
    D --> E["Verify: Av = λv"]
    
    F["Geometric Interpretation:"] --> G["Eigenvector v:<br/>Direction unchanged<br/>by transformation A"]
    G --> H["Eigenvalue λ:<br/>Scaling factor<br/>along eigenvector"]
    
    I["Applications:"] --> J["Principal Component Analysis<br/>(PCA)"]
    I --> K["Neural Network Dynamics"]
    I --> L["Matrix Decomposition"]
    
    style A fill:#e1f5ff
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style D fill:#e8f5e9
    style E fill:#ffebee
    style F fill:#e0f2f1
    style G fill:#e0f2f1
    style H fill:#e0f2f1
    style I fill:#fff9c4
    style J fill:#fff9c4
    style K fill:#fff9c4
    style L fill:#fff9c4
```

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

**Eigenvalue Decomposition Visualization:**

```python
# Visualize eigenvalue decomposition
A = np.array([[3, 1], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)

# Create a circle of vectors
theta = np.linspace(0, 2*np.pi, 100)
circle_vectors = np.array([np.cos(theta), np.sin(theta)])

# Transform circle by matrix A
transformed_circle = A @ circle_vectors

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Original circle
axes[0].plot(circle_vectors[0], circle_vectors[1], 'b-', linewidth=2, label='Unit Circle')
axes[0].set_title('Original Unit Circle', fontsize=12, fontweight='bold')
axes[0].set_xlabel('X', fontsize=11)
axes[0].set_ylabel('Y', fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_aspect('equal')
axes[0].legend(fontsize=9)

# Transformed ellipse
axes[1].plot(transformed_circle[0], transformed_circle[1], 'r-', linewidth=2, label='Transformed Ellipse')

# Draw eigenvectors
for i, (eigenval, eigenvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
    # Scale eigenvector by eigenvalue
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
plt.savefig('docs/images/eigenvalue_decomposition.png', dpi=150, bbox_inches='tight')
plt.show()
```

## Linear Transformations

**Mermaid Diagram: Types of Linear Transformations**

```mermaid
graph TB
    A["Linear Transformation<br/>y = Ax"] --> B["Identity<br/>No change"]
    A --> C["Rotation<br/>Preserves length"]
    A --> D["Reflection<br/>Mirror image"]
    A --> E["Scaling<br/>Stretch/shrink"]
    A --> F["Shear<br/>Skew shape"]
    
    B --> G["Matrix: [[1,0],[0,1]]<br/>Det = 1"]
    C --> H["Matrix: [[cos θ, -sin θ],<br/>           [sin θ, cos θ]]<br/>Det = 1"]
    D --> I["Matrix: [[1,0],[0,-1]]<br/>Det = -1"]
    E --> J["Matrix: [[sx,0],[0,sy]]<br/>Det = sx × sy"]
    F --> K["Matrix: [[1,k],[0,1]]<br/>Det = 1"]
    
    style A fill:#e1f5ff
    style B fill:#e8f5e9
    style C fill:#fff3e0
    style D fill:#ffebee
    style E fill:#f3e5f5
    style F fill:#e0f2f1
```

### Common Matrix Transformations

```python
# Visualize common linear transformations
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

# Original
axes[0].plot(unit_square[0], unit_square[1], 'b-', linewidth=2, marker='o', markersize=4)
axes[0].fill(unit_square[0], unit_square[1], alpha=0.3, color='blue')
axes[0].set_title('Original Unit Square', fontsize=11, fontweight='bold')
axes[0].set_xlim(-2, 3)
axes[0].set_ylim(-2, 3)
axes[0].grid(True, alpha=0.3)
axes[0].set_aspect('equal')

# Apply each transformation
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
plt.savefig('docs/images/linear_transformations.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Mermaid Diagram: Matrix Rank Concept**

```mermaid
flowchart TD
    A["Matrix A"] --> B["Column Vectors"]
    B --> C{"Are columns<br/>linearly independent?"}
    
    C -->|"All independent"| D["Full Rank<br/>Rank = min(m,n)"]
    C -->|"Some dependent"| E["Low Rank<br/>Rank < min(m,n)"]
    
    D --> F["Column Space =<br/>Full dimension"]
    E --> G["Column Space =<br/>Reduced dimension"]
    
    H["Rank Properties:"] --> I["Rank(A) = Rank(Aᵀ)"]
    H --> J["Rank(AB) ≤ min(Rank(A), Rank(B))"]
    H --> K["Full rank ⟺ det(A) ≠ 0<br/>(for square matrices)"]
    
    style A fill:#e1f5ff
    style B fill:#fff3e0
    style C fill:#fff3e0
    style D fill:#e8f5e9
    style E fill:#ffebee
    style F fill:#e8f5e9
    style G fill:#ffebee
    style H fill:#f3e5f5
    style I fill:#f3e5f5
    style J fill:#f3e5f5
    style K fill:#f3e5f5
```

### Matrix Rank Visualization

```python
# Visualize matrix rank (dimensionality of column space)
A_full_rank = np.array([[2, 1], [1, 2]])  # Rank 2
A_low_rank = np.array([[2, 4], [1, 2]])   # Rank 1 (columns are linearly dependent)

# Column vectors
col1_full = A_full_rank[:, 0]
col2_full = A_full_rank[:, 1]
col1_low = A_low_rank[:, 0]
col2_low = A_low_rank[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Full rank
axes[0].arrow(0, 0, col1_full[0], col1_full[1], head_width=0.2, head_length=0.2,
            fc='blue', ec='blue', linewidth=2, label='Column 1')
axes[0].arrow(0, 0, col2_full[0], col2_full[1], head_width=0.2, head_length=0.2,
            fc='red', ec='red', linewidth=2, label='Column 2')
axes[0].set_title(f'Full Rank Matrix (Rank = 2)\nColumns are independent', 
                 fontsize=12, fontweight='bold')
axes[0].set_xlim(-1, 3)
axes[0].set_ylim(-1, 3)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=9)
axes[0].set_aspect('equal')

# Low rank
axes[1].arrow(0, 0, col1_low[0], col1_low[1], head_width=0.2, head_length=0.2,
            fc='blue', ec='blue', linewidth=2, label='Column 1')
axes[1].arrow(0, 0, col2_low[0], col2_low[1], head_width=0.2, head_length=0.2,
            fc='red', ec='red', linewidth=2, label='Column 2 (2×Col1)')
axes[1].set_title(f'Low Rank Matrix (Rank = 1)\nColumns are dependent', 
                 fontsize=12, fontweight='bold')
axes[1].set_xlim(-1, 5)
axes[1].set_ylim(-1, 3)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=9)
axes[1].set_aspect('equal')

plt.tight_layout()
plt.savefig('docs/images/matrix_rank.png', dpi=150, bbox_inches='tight')
plt.show()
```

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

**Mermaid Diagram: Linear Combination Concept**

```mermaid
flowchart LR
    A["Basis Vector e₁"] --> D["Linear Combination"]
    B["Basis Vector e₂"] --> D
    C["Scalars c₁, c₂"] --> D
    D --> E["Result Vector<br/>v = c₁e₁ + c₂e₂"]
    
    F["Properties:"] --> G["Span: All possible<br/>linear combinations"]
    F --> H["Basis: Minimal set of<br/>linearly independent vectors"]
    F --> I["Dimension: Number of<br/>vectors in basis"]
    
    style A fill:#e1f5ff
    style B fill:#e8f5e9
    style C fill:#fff3e0
    style D fill:#f3e5f5
    style E fill:#ffebee
    style F fill:#e0f2f1
    style G fill:#e0f2f1
    style H fill:#e0f2f1
    style I fill:#e0f2f1
```

### Linear Combinations

```python
# Visualize linear combinations of basis vectors
e1 = np.array([1, 0])
e2 = np.array([0, 1])

# Different linear combinations
combinations = [
    (2, 1, '2e₁ + e₂'),
    (1, 2, 'e₁ + 2e₂'),
    (-1, 1, '-e₁ + e₂'),
    (1, -1, 'e₁ - e₂')
]

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for idx, (c1, c2, label) in enumerate(combinations):
    result = c1 * e1 + c2 * e2
    
    # Basis vectors
    axes[idx].arrow(0, 0, e1[0], e1[1], head_width=0.15, head_length=0.15,
                   fc='blue', ec='blue', linewidth=2, alpha=0.5, label='e₁')
    axes[idx].arrow(0, 0, e2[0], e2[1], head_width=0.15, head_length=0.15,
                   fc='green', ec='green', linewidth=2, alpha=0.5, label='e₂')
    
    # Scaled basis vectors
    axes[idx].arrow(0, 0, c1*e1[0], c1*e1[1], head_width=0.1, head_length=0.1,
                   fc='blue', ec='blue', linewidth=1.5, linestyle='--', alpha=0.7)
    axes[idx].arrow(c1*e1[0], c1*e1[1], c2*e2[0], c2*e2[1], 
                   head_width=0.1, head_length=0.1,
                   fc='green', ec='green', linewidth=1.5, linestyle='--', alpha=0.7)
    
    # Result
    axes[idx].arrow(0, 0, result[0], result[1], head_width=0.2, head_length=0.2,
                   fc='red', ec='red', linewidth=3, label=label)
    
    axes[idx].set_xlim(-2, 3)
    axes[idx].set_ylim(-2, 3)
    axes[idx].set_title(label, fontsize=11, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].axhline(y=0, color='k', linewidth=0.5)
    axes[idx].axvline(x=0, color='k', linewidth=0.5)
    axes[idx].legend(fontsize=8)
    axes[idx].set_aspect('equal')

plt.tight_layout()
plt.savefig('docs/images/linear_combinations.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Mermaid Diagram: Orthogonality and Orthonormality**

```mermaid
graph TB
    A["Two Vectors v₁ and v₂"] --> B{"Check Dot Product<br/>v₁ · v₂ = ?"}
    
    B -->|"= 0"| C["Orthogonal<br/>(Perpendicular)"]
    B -->|"≠ 0"| D["Not Orthogonal"]
    
    C --> E{"Are they also<br/>unit vectors?"}
    E -->|"Yes: ||v₁|| = ||v₂|| = 1"| F["Orthonormal"]
    E -->|"No"| G["Just Orthogonal"]
    
    H["Orthogonal Basis:"] --> I["All vectors are<br/>mutually orthogonal"]
    I --> J["Easier to work with<br/>in computations"]
    
    style A fill:#e1f5ff
    style B fill:#fff3e0
    style C fill:#e8f5e9
    style D fill:#ffebee
    style E fill:#f3e5f5
    style F fill:#e8f5e9
    style G fill:#fff3e0
    style H fill:#e0f2f1
    style I fill:#e0f2f1
    style J fill:#e0f2f1
```

### Orthogonal Vectors

```python
# Visualize orthogonal (perpendicular) vectors
v1 = np.array([3, 0])
v2 = np.array([0, 2])

# Check orthogonality
dot_product = np.dot(v1, v2)
is_orthogonal = abs(dot_product) < 1e-10

fig, ax = plt.subplots(figsize=(8, 8))

ax.arrow(0, 0, v1[0], v1[1], head_width=0.2, head_length=0.2,
        fc='blue', ec='blue', linewidth=3, label=f'v1 = {v1}')
ax.arrow(0, 0, v2[0], v2[1], head_width=0.2, head_length=0.2,
        fc='red', ec='red', linewidth=3, label=f'v2 = {v2}')

# Draw right angle indicator
ax.plot([0.3, 0.3, 0], [0, 0.2, 0.2], 'k-', linewidth=1.5)

ax.set_xlim(-1, 4)
ax.set_ylim(-1, 3)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title(f'Orthogonal Vectors\nv1 · v2 = {dot_product} (should be 0)', 
            fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.legend(fontsize=10)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('docs/images/orthogonal_vectors.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Matrix Multiplication Step-by-Step

```python
# Detailed visualization of matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A @ B

fig = plt.figure(figsize=(16, 10))

# Create a detailed visualization
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# Matrix A
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(A, cmap='Blues', aspect='auto', vmin=0, vmax=10)
ax1.set_title('Matrix A', fontsize=12, fontweight='bold')
for i in range(2):
    for j in range(2):
        ax1.text(j, i, A[i, j], ha='center', va='center', fontsize=14, 
                fontweight='bold', color='white')
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['Col 1', 'Col 2'])
ax1.set_yticklabels(['Row 1', 'Row 2'])

# Matrix B
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(B, cmap='Greens', aspect='auto', vmin=0, vmax=10)
ax2.set_title('Matrix B', fontsize=12, fontweight='bold')
for i in range(2):
    for j in range(2):
        ax2.text(j, i, B[i, j], ha='center', va='center', fontsize=14, 
                fontweight='bold', color='white')
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Col 1', 'Col 2'])
ax2.set_yticklabels(['Row 1', 'Row 2'])

# Result C
ax3 = fig.add_subplot(gs[0, 2:])
im3 = ax3.imshow(C, cmap='Reds', aspect='auto', vmin=0, vmax=50)
ax3.set_title('Result: A × B', fontsize=12, fontweight='bold')
for i in range(2):
    for j in range(2):
        ax3.text(j, i, C[i, j], ha='center', va='center', fontsize=14, 
                fontweight='bold', color='white')
ax3.set_xticks([0, 1])
ax3.set_yticks([0, 1])
ax3.set_xticklabels(['Col 1', 'Col 2'])
ax3.set_yticklabels(['Row 1', 'Row 2'])

# Step-by-step calculations
ax4 = fig.add_subplot(gs[1:, :])
ax4.axis('off')
calculation_text = f"""
Matrix Multiplication Step-by-Step:

C[0,0] = A[0,:] · B[:,0] = [{A[0,0]}, {A[0,1]}] · [{B[0,0]}, {B[1,0]}] = {A[0,0]}×{B[0,0]} + {A[0,1]}×{B[1,0]} = {C[0,0]}

C[0,1] = A[0,:] · B[:,1] = [{A[0,0]}, {A[0,1]}] · [{B[0,1]}, {B[1,1]}] = {A[0,0]}×{B[0,1]} + {A[0,1]}×{B[1,1]} = {C[0,1]}

C[1,0] = A[1,:] · B[:,0] = [{A[1,0]}, {A[1,1]}] · [{B[0,0]}, {B[1,0]}] = {A[1,0]}×{B[0,0]} + {A[1,1]}×{B[1,0]} = {C[1,0]}

C[1,1] = A[1,:] · B[:,1] = [{A[1,0]}, {A[1,1]}] · [{B[0,1]}, {B[1,1]}] = {A[1,0]}×{B[0,1]} + {A[1,1]}×{B[1,1]} = {C[1,1]}

Key: Each element C[i,j] is the dot product of row i of A and column j of B
"""
ax4.text(0.1, 0.5, calculation_text, fontsize=11, family='monospace',
        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.savefig('docs/images/matrix_multiplication_detailed.png', dpi=150, bbox_inches='tight')
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

## Generated Visualizations

This document includes code to generate the following visualizations:

1. **Vector Operations:**
   - Vector addition with parallelogram rule
   - Dot product with angle visualization
   - Vector projection
   - Cross product in 3D
   - Vector magnitude (norm)
   - Scalar multiplication

2. **Matrix Operations:**
   - Matrix multiplication heatmaps
   - Matrix transpose visualization
   - Determinant geometric interpretation (area scaling)
   - Matrix inverse transformation
   - Step-by-step matrix multiplication

3. **Linear Transformations:**
   - Identity, rotation, reflection, scaling, shear
   - Matrix rank visualization
   - Eigenvalue decomposition (circle to ellipse)
   - Eigenvectors and transformations

4. **Vector Spaces:**
   - 2D vector space with basis vectors
   - Linear combinations
   - Orthogonal vectors

All visualizations are saved to `docs/images/` when you run the code examples. These graphs help visualize abstract linear algebra concepts and make them easier to understand!

---

## Linear Algebra in Image Processing

Linear algebra is fundamental to image processing. Images are represented as matrices, and most image operations are matrix operations. This section demonstrates practical applications of linear algebra concepts to real image processing tasks.

### Image Representation as Matrices

**Key Concept**: Images are 2D or 3D matrices where:
- **Grayscale images**: 2D matrix (height × width), each element is a pixel intensity (0-255)
- **Color images**: 3D matrix (height × width × channels), typically RGB (3 channels)

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

# Create a sample grayscale image (matrix)
def create_sample_image():
    """Create a simple test image as a matrix"""
    img = np.zeros((100, 100), dtype=np.uint8)
    
    # Draw geometric shapes
    # Rectangle
    img[20:60, 20:60] = 255
    
    # Circle
    y, x = np.ogrid[:100, :100]
    center_x, center_y = 70, 70
    mask = (x - center_x)**2 + (y - center_y)**2 <= 20**2
    img[mask] = 180
    
    # Diagonal line
    for i in range(100):
        if 0 <= i < 100:
            img[i, i] = 200
    
    return img

# Display image as matrix
img = create_sample_image()
print(f"Image shape (matrix dimensions): {img.shape}")
print(f"Image data type: {img.dtype}")
print(f"Pixel value range: {img.min()} to {img.max()}")
print(f"\nFirst 10x10 pixels (matrix values):")
print(img[:10, :10])

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Image Visualization', fontsize=14, fontweight='bold')
axes[0].axis('off')

# Show as matrix heatmap
im = axes[1].imshow(img, cmap='gray', aspect='auto')
axes[1].set_title('Image as Matrix (Pixel Values)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Width (columns)', fontsize=12)
axes[1].set_ylabel('Height (rows)', fontsize=12)
plt.colorbar(im, ax=axes[1], label='Pixel Intensity')

plt.tight_layout()
plt.savefig('docs/images/image_as_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Image Transformations Using Matrix Operations

**Rotation, Scaling, Reflection, and Translation** can all be represented as matrix multiplications.

```python
def apply_rotation_matrix(image, angle_degrees):
    """Rotate image using rotation matrix"""
    angle_rad = np.radians(angle_degrees)
    # Rotation matrix
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                  [np.sin(angle_rad), np.cos(angle_rad)]])
    
    # Apply rotation using scipy (handles interpolation)
    rotated = ndimage.rotate(image, angle_degrees, reshape=False, order=1)
    return rotated, R

def apply_scaling_matrix(image, scale_x, scale_y):
    """Scale image using scaling matrix"""
    S = np.array([[scale_x, 0],
                  [0, scale_y]])
    
    # Apply scaling
    h, w = image.shape
    new_h, new_w = int(h * scale_y), int(w * scale_x)
    scaled = ndimage.zoom(image, (scale_y, scale_x), order=1)
    return scaled, S

def apply_reflection_matrix(image, axis='x'):
    """Reflect image using reflection matrix"""
    if axis == 'x':
        R = np.array([[1, 0], [0, -1]])  # Reflect across x-axis
        reflected = np.flipud(image)
    elif axis == 'y':
        R = np.array([[-1, 0], [0, 1]])  # Reflect across y-axis
        reflected = np.fliplr(image)
    else:  # y=x line
        R = np.array([[0, 1], [1, 0]])
        reflected = np.transpose(image)
    
    return reflected, R

# Create test image
img = create_sample_image()

# Apply transformations
rotated, R_rot = apply_rotation_matrix(img, 45)
scaled, S_scale = apply_scaling_matrix(img, 1.5, 1.5)
reflected_x, R_ref_x = apply_reflection_matrix(img, 'x')
reflected_y, R_ref_y = apply_reflection_matrix(img, 'y')

# Visualize transformations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

transformations = [
    ('Original', img, np.eye(2)),
    ('Rotation 45°', rotated, R_rot),
    ('Scaling 1.5x', scaled, S_scale),
    ('Reflection (x-axis)', reflected_x, R_ref_x),
    ('Reflection (y-axis)', reflected_y, R_ref_y),
]

for idx, (name, transformed_img, T) in enumerate(transformations):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    ax.imshow(transformed_img, cmap='gray')
    det = np.linalg.det(T)
    ax.set_title(f'{name}\nMatrix: {T}\nDet = {det:.2f}', 
                fontsize=11, fontweight='bold')
    ax.axis('off')

# Hide last subplot
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('docs/images/image_transformations_linear_algebra.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nKey Observations:")
print("- Rotation matrix: Preserves distances, det = 1")
print("- Scaling matrix: Changes area by det(T) factor")
print("- Reflection matrix: Reverses orientation, det = -1")
```

### Image Filtering Using Convolution (Matrix Multiplication)

**Convolution** is a fundamental image processing operation, implemented as matrix multiplication with filters/kernels.

```python
def apply_convolution_filter(image, kernel):
    """Apply convolution filter to image (edge detection, blur, etc.)"""
    # Convolution is essentially matrix multiplication with a sliding window
    filtered = ndimage.convolve(image.astype(float), kernel, mode='constant')
    return filtered.astype(np.uint8)

# Define common filters (kernels) as matrices
filters = {
    'Identity (No change)': np.array([[0, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 0]]),
    
    'Edge Detection (Sobel X)': np.array([[-1, 0, 1],
                                           [-2, 0, 2],
                                           [-1, 0, 1]]),
    
    'Edge Detection (Sobel Y)': np.array([[-1, -2, -1],
                                           [0, 0, 0],
                                           [1, 2, 1]]),
    
    'Blur (Gaussian-like)': np.array([[1, 2, 1],
                                     [2, 4, 2],
                                     [1, 2, 1]]) / 16,
    
    'Sharpen': np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]]),
    
    'Emboss': np.array([[-2, -1, 0],
                       [-1, 1, 1],
                       [0, 1, 2]])
}

# Apply filters
img = create_sample_image()
results = {}

for name, kernel in filters.items():
    filtered = apply_convolution_filter(img, kernel)
    # Normalize for display
    if filtered.min() < 0:
        filtered = filtered - filtered.min()
    if filtered.max() > 255:
        filtered = (filtered / filtered.max() * 255).astype(np.uint8)
    results[name] = filtered

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (name, filtered_img) in enumerate(results.items()):
    axes[idx].imshow(filtered_img, cmap='gray')
    axes[idx].set_title(f'{name}', fontsize=11, fontweight='bold')
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('docs/images/image_filtering_convolution.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nConvolution as Matrix Operations:")
print("- Each filter is a small matrix (kernel)")
print("- Convolution = sliding window matrix multiplication")
print("- Different kernels extract different features (edges, blur, etc.)")
```

### Image Compression Using Matrix Decomposition

**Singular Value Decomposition (SVD)** can compress images by keeping only the most important components.

```python
def compress_image_svd(image, k_components):
    """Compress image using SVD (Singular Value Decomposition)"""
    # SVD: A = U × Σ × V^T
    U, s, Vt = np.linalg.svd(image.astype(float), full_matrices=False)
    
    # Keep only top k components
    U_k = U[:, :k_components]
    s_k = s[:k_components]
    Vt_k = Vt[:k_components, :]
    
    # Reconstruct: A ≈ U_k × diag(s_k) × Vt_k
    compressed = U_k @ np.diag(s_k) @ Vt_k
    
    # Calculate compression ratio
    original_size = image.size
    compressed_size = U_k.size + s_k.size + Vt_k.size
    compression_ratio = original_size / compressed_size
    
    return compressed.astype(np.uint8), compression_ratio, s

# Create a more complex test image
def create_complex_image():
    """Create image with more detail"""
    img = np.zeros((200, 200), dtype=np.uint8)
    
    # Add multiple shapes
    img[50:150, 50:150] = 255  # Large square
    img[20:80, 20:80] = 180    # Small square
    
    # Add gradient
    for i in range(200):
        img[i, :] = int(255 * i / 200)
    
    # Add circles
    y, x = np.ogrid[:200, :200]
    center_x, center_y = 100, 100
    mask = (x - center_x)**2 + (y - center_y)**2 <= 40**2
    img[mask] = 150
    
    return img

img = create_complex_image()

# Compress with different numbers of components
k_values = [5, 10, 20, 50, 100]
compressions = {}

for k in k_values:
    compressed, ratio, singular_values = compress_image_svd(img, k)
    compressions[k] = (compressed, ratio, singular_values)

# Visualize compression results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Original
axes[0].imshow(img, cmap='gray')
axes[0].set_title(f'Original\nSize: {img.size} pixels', 
                 fontsize=11, fontweight='bold')
axes[0].axis('off')

# Compressed versions
for idx, k in enumerate(k_values, 1):
    compressed, ratio, s = compressions[k]
    axes[idx].imshow(compressed, cmap='gray')
    axes[idx].set_title(f'k={k} components\nCompression: {ratio:.2f}x\nMSE: {np.mean((img - compressed)**2):.1f}', 
                       fontsize=10, fontweight='bold')
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('docs/images/image_compression_svd.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot singular values (importance)
fig, ax = plt.subplots(figsize=(10, 6))
_, _, s_full = compress_image_svd(img, min(img.shape))
ax.plot(s_full[:50], 'b-o', linewidth=2, markersize=4)
ax.set_xlabel('Component Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Singular Value', fontsize=12, fontweight='bold')
ax.set_title('Singular Values (Importance of Components)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')
plt.tight_layout()
plt.savefig('docs/images/singular_values_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nSVD Image Compression:")
print("- Images can be represented as: Image = U × Σ × V^T")
print("- Keep only top k singular values for compression")
print("- Higher k = better quality, lower compression")
print("- Lower k = more compression, lower quality")
```

### Edge Detection Using Matrix Operations

**Edge detection** uses gradient computation, which is a matrix operation.

```python
def edge_detection_sobel(image):
    """Edge detection using Sobel operators (matrix-based)"""
    # Sobel X kernel (detects vertical edges)
    sobel_x = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    
    # Sobel Y kernel (detects horizontal edges)
    sobel_y = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
    
    # Apply convolution
    edges_x = ndimage.convolve(image.astype(float), sobel_x, mode='constant')
    edges_y = ndimage.convolve(image.astype(float), sobel_y, mode='constant')
    
    # Magnitude of gradient: ||∇I|| = sqrt((∂I/∂x)² + (∂I/∂y)²)
    edges_magnitude = np.sqrt(edges_x**2 + edges_y**2)
    
    # Direction of gradient: θ = atan2(∂I/∂y, ∂I/∂x)
    edges_direction = np.arctan2(edges_y, edges_x)
    
    return edges_x, edges_y, edges_magnitude, edges_direction

img = create_complex_image()
edges_x, edges_y, edges_mag, edges_dir = edge_detection_sobel(img)

# Normalize for display
edges_x_norm = ((edges_x - edges_x.min()) / (edges_x.max() - edges_x.min()) * 255).astype(np.uint8)
edges_y_norm = ((edges_y - edges_y.min()) / (edges_y.max() - edges_y.min()) * 255).astype(np.uint8)
edges_mag_norm = ((edges_mag - edges_mag.min()) / (edges_mag.max() - edges_mag.min()) * 255).astype(np.uint8)

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(edges_x_norm, cmap='gray')
axes[0, 1].set_title('Vertical Edges (Sobel X)\n∂I/∂x', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

axes[1, 0].imshow(edges_y_norm, cmap='gray')
axes[1, 0].set_title('Horizontal Edges (Sobel Y)\n∂I/∂y', fontsize=12, fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(edges_mag_norm, cmap='gray')
axes[1, 1].set_title('Edge Magnitude\n||∇I|| = √((∂I/∂x)² + (∂I/∂y)²)', 
                     fontsize=12, fontweight='bold')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('docs/images/edge_detection_matrix_operations.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nEdge Detection with Linear Algebra:")
print("- Gradient = (∂I/∂x, ∂I/∂y) - a vector!")
print("- Edge magnitude = ||gradient|| = matrix norm")
print("- Edge direction = angle of gradient vector")
print("- All computed using matrix operations (convolution)")
```

### Image Enhancement Using Matrix Operations

**Enhancement operations** like brightness, contrast, and histogram equalization use matrix operations.

```python
def enhance_brightness(image, factor):
    """Adjust brightness: I_new = I + factor (matrix addition)"""
    enhanced = np.clip(image.astype(float) + factor, 0, 255).astype(np.uint8)
    return enhanced

def enhance_contrast(image, factor):
    """Adjust contrast: I_new = factor × (I - mean) + mean (matrix scaling)"""
    mean = image.mean()
    enhanced = np.clip(factor * (image.astype(float) - mean) + mean, 0, 255).astype(np.uint8)
    return enhanced

def enhance_histogram_equalization(image):
    """Histogram equalization using cumulative distribution"""
    # Compute histogram (frequency of each pixel value)
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    
    # Cumulative distribution function (CDF)
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    
    # Map pixel values using CDF (lookup table = matrix transformation)
    enhanced = cdf_normalized[image].astype(np.uint8)
    return enhanced, hist, cdf

img = create_complex_image()

# Apply enhancements
bright = enhance_brightness(img, 50)
dark = enhance_brightness(img, -50)
high_contrast = enhance_contrast(img, 1.5)
low_contrast = enhance_contrast(img, 0.7)
equalized, hist, cdf = enhance_histogram_equalization(img)

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

enhancements = [
    ('Original', img),
    ('Brightness +50', bright),
    ('Brightness -50', dark),
    ('High Contrast (1.5x)', high_contrast),
    ('Low Contrast (0.7x)', low_contrast),
    ('Histogram Equalized', equalized)
]

for idx, (name, enhanced_img) in enumerate(enhancements):
    axes[idx].imshow(enhanced_img, cmap='gray')
    axes[idx].set_title(name, fontsize=11, fontweight='bold')
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('docs/images/image_enhancement_matrix_ops.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot histogram and CDF
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(256), hist, color='steelblue', alpha=0.7)
axes[0].set_xlabel('Pixel Value', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0].set_title('Original Image Histogram', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].plot(cdf, linewidth=2, color='green')
axes[1].set_xlabel('Pixel Value', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Cumulative Frequency', fontsize=12, fontweight='bold')
axes[1].set_title('Cumulative Distribution Function (CDF)', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('docs/images/histogram_equalization.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nImage Enhancement with Linear Algebra:")
print("- Brightness: I_new = I + c (matrix addition)")
print("- Contrast: I_new = α × I + β (matrix scaling + addition)")
print("- Histogram equalization: Lookup table transformation")
print("- All are matrix/vector operations!")
```

### Color Image Processing

**Color images** are 3D matrices (height × width × channels). Each channel can be processed independently.

```python
def create_color_image():
    """Create a color test image (3D matrix)"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Red channel
    img[20:60, 20:60, 0] = 255
    # Green channel
    img[40:80, 40:80, 1] = 255
    # Blue channel
    y, x = np.ogrid[:100, :100]
    center_x, center_y = 70, 30
    mask = (x - center_x)**2 + (y - center_y)**2 <= 15**2
    img[mask, 2] = 255
    
    return img

def convert_to_grayscale(image):
    """Convert color to grayscale: Gray = 0.299×R + 0.587×G + 0.114×B (weighted sum)"""
    # This is a matrix multiplication: [R, G, B] × [0.299, 0.587, 0.114]^T
    weights = np.array([0.299, 0.587, 0.114])
    grayscale = np.dot(image, weights).astype(np.uint8)
    return grayscale

def adjust_color_channel(image, channel, factor):
    """Adjust individual color channel (matrix scaling)"""
    adjusted = image.copy().astype(float)
    adjusted[:, :, channel] *= factor
    return np.clip(adjusted, 0, 255).astype(np.uint8)

# Create color image
color_img = create_color_image()

# Process color image
grayscale = convert_to_grayscale(color_img)
red_boost = adjust_color_channel(color_img, 0, 1.5)
green_boost = adjust_color_channel(color_img, 1, 1.5)
blue_boost = adjust_color_channel(color_img, 2, 1.5)

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

images = [
    ('Original Color', color_img),
    ('Grayscale\n(Weighted Sum)', grayscale),
    ('Red Channel Boosted', red_boost),
    ('Green Channel Boosted', green_boost),
    ('Blue Channel Boosted', blue_boost),
]

for idx, (name, img) in enumerate(images):
    if len(img.shape) == 3:
        axes[idx].imshow(img)
    else:
        axes[idx].imshow(img, cmap='gray')
    axes[idx].set_title(name, fontsize=11, fontweight='bold')
    axes[idx].axis('off')

axes[5].axis('off')

plt.tight_layout()
plt.savefig('docs/images/color_image_processing.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nColor Image Processing:")
print("- Color images = 3D matrices (H × W × 3)")
print("- Each channel (R, G, B) can be processed independently")
print("- Grayscale conversion = weighted matrix multiplication")
print("- Channel adjustment = matrix scaling per channel")
```

### Summary: Linear Algebra in Image Processing

**Key Takeaways**:

1. **Images are Matrices**: Every image is a 2D or 3D matrix
2. **Transformations = Matrix Multiplication**: Rotation, scaling, reflection
3. **Filtering = Convolution**: Matrix multiplication with kernels
4. **Compression = Matrix Decomposition**: SVD for efficient storage
5. **Enhancement = Matrix Operations**: Addition, scaling, lookup tables
6. **Edge Detection = Gradient Computation**: Vector operations on matrices

**Mermaid Diagram: Linear Algebra Operations in Image Processing**

```mermaid
flowchart TD
    A["Image<br/>(Matrix)"] --> B["Transformations"]
    A --> C["Filtering"]
    A --> D["Compression"]
    A --> E["Enhancement"]
    
    B --> B1["Rotation: R × Image"]
    B --> B2["Scaling: S × Image"]
    B --> B3["Reflection: M × Image"]
    
    C --> C1["Convolution: Kernel × Image"]
    C --> C2["Edge Detection: Gradient"]
    C --> C3["Blur/Sharpen: Filter Matrix"]
    
    D --> D1["SVD: U × Σ × V^T"]
    D --> D2["Keep top k components"]
    D --> D3["Reconstruct: U_k × Σ_k × V_k^T"]
    
    E --> E1["Brightness: I + c"]
    E --> E2["Contrast: α × I + β"]
    E --> E3["Histogram: Lookup Table"]
    
    style A fill:#e1f5ff
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style D fill:#e8f5e9
    style E fill:#ffebee
```

**Practice Exercises**:

1. Implement image rotation from scratch using rotation matrices
2. Create custom filters for edge detection
3. Compress an image using SVD and compare quality vs compression ratio
4. Implement histogram equalization manually
5. Apply multiple transformations in sequence (matrix multiplication chain)

**Next Steps**: See [Applied Linear Algebra Projects - Image Transformations](projects/01_applied_linear_algebra_projects.md#project-1-image-transformations) for more advanced image processing projects!

---

All visualizations are saved to `docs/images/` when you run the code examples. These graphs help visualize abstract linear algebra concepts and make them easier to understand!

## Next Steps

- Study [Calculus for Deep Learning](02_calculus.md)
- Learn about [Neural Networks](04_neural_networks.md)
- **Practice with [Applied Projects](../projects/01_applied_linear_algebra_projects.md)** - Hands-on projects using linear algebra!
