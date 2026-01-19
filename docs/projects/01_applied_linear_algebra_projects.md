# Applied Linear Algebra Projects

## Table of Contents
1. [Project 1: Image Transformations](#project-1-image-transformations)
2. [Project 2: Principal Component Analysis (PCA)](#project-2-principal-component-analysis-pca)
3. [Project 3: Linear Regression from Scratch](#project-3-linear-regression-from-scratch)
4. [Project 4: Simple Neural Network](#project-4-simple-neural-network)
5. [Project 5: Data Preprocessing Pipeline](#project-5-data-preprocessing-pipeline)
6. [Project 6: Face Recognition with Eigenfaces](#project-6-face-recognition-with-eigenfaces)

## Project 1: Image Transformations

**Objective**: Apply matrix transformations to rotate, scale, and reflect images.

### Theory
Images are represented as matrices where each pixel is an element. Linear transformations can be applied using matrix multiplication.

### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches

def create_test_image(size=100):
    """Create a simple test image"""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # Draw a rectangle
    img[20:60, 20:60] = [255, 0, 0]  # Red square
    return img

def apply_transformation(image, transformation_matrix):
    """Apply linear transformation to image"""
    h, w = image.shape[:2]
    
    # Create coordinate grid
    y, x = np.mgrid[0:h, 0:w]
    coords = np.array([x.flatten(), y.flatten(), np.ones(x.size)])
    
    # Apply transformation (2D, so we use 2x2 matrix)
    # For 2D, we only need 2x2, but we'll use homogeneous coordinates
    if transformation_matrix.shape == (2, 2):
        T = np.vstack([np.hstack([transformation_matrix, np.zeros((2, 1))]), 
                      np.array([0, 0, 1])])
    else:
        T = transformation_matrix
    
    # Transform coordinates
    new_coords = T @ coords
    new_x = new_coords[0].reshape(h, w)
    new_y = new_coords[1].reshape(h, w)
    
    # Create output image
    output = np.zeros_like(image)
    
    # Map pixels (simplified - for production use scipy.ndimage)
    for i in range(h):
        for j in range(w):
            nx, ny = int(new_x[i, j]), int(new_y[i, j])
            if 0 <= nx < w and 0 <= ny < h:
                output[ny, nx] = image[i, j]
    
    return output

# Create test image
img = create_test_image(100)

# Define transformations
transformations = {
    'Original': np.eye(2),
    'Rotation 45°': np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                             [np.sin(np.pi/4), np.cos(np.pi/4)]]),
    'Scaling 2x': np.array([[2, 0], [0, 2]]),
    'Reflection (y-axis)': np.array([[-1, 0], [0, 1]]),
    'Shear': np.array([[1, 0.5], [0, 1]])
}

# Visualize transformations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (name, T) in enumerate(transformations.items()):
    if name == 'Original':
        transformed = img
    else:
        transformed = apply_transformation(img, T)
    
    axes[idx].imshow(transformed)
    axes[idx].set_title(f'{name}\nDet = {np.linalg.det(T):.2f}', 
                       fontsize=12, fontweight='bold')
    axes[idx].axis('off')

axes[5].axis('off')  # Hide last subplot
plt.tight_layout()
plt.savefig('docs/images/image_transformations.png', dpi=150, bbox_inches='tight')
plt.show()

print("Image Transformations Project Complete!")
print("Key Concepts:")
print("- Images as matrices")
print("- Matrix transformations")
print("- Determinant as area scaling factor")
```

## Project 2: Principal Component Analysis (PCA)

**Objective**: Reduce dimensionality of data using eigenvalues and eigenvectors.

### Theory
PCA finds the directions (eigenvectors) of maximum variance in data and projects data onto these directions.

### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate sample data
np.random.seed(42)
X, y = make_blobs(n_samples=200, centers=3, n_features=2, random_state=42)

# Center the data
X_centered = X - np.mean(X, axis=0)

# Calculate covariance matrix
cov_matrix = np.cov(X_centered.T)

# Find eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort by eigenvalue (descending)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Principal components
PC1 = eigenvectors[:, 0]  # First principal component
PC2 = eigenvectors[:, 1]  # Second principal component

# Project data onto principal components
X_projected = X_centered @ eigenvectors

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Original data with principal components
axes[0].scatter(X_centered[:, 0], X_centered[:, 1], c=y, cmap='viridis', alpha=0.6)
axes[0].arrow(0, 0, PC1[0]*3, PC1[1]*3, head_width=0.3, head_length=0.3,
             fc='red', ec='red', linewidth=3, label=f'PC1 (λ={eigenvalues[0]:.2f})')
axes[0].arrow(0, 0, PC2[0]*3, PC2[1]*3, head_width=0.3, head_length=0.3,
             fc='blue', ec='blue', linewidth=3, label=f'PC2 (λ={eigenvalues[1]:.2f})')
axes[0].set_xlabel('Feature 1', fontsize=12)
axes[0].set_ylabel('Feature 2', fontsize=12)
axes[0].set_title('Original Data with Principal Components', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_aspect('equal')

# Projected data
axes[1].scatter(X_projected[:, 0], X_projected[:, 1], c=y, cmap='viridis', alpha=0.6)
axes[1].set_xlabel('Principal Component 1', fontsize=12)
axes[1].set_ylabel('Principal Component 2', fontsize=12)
axes[1].set_title('Data Projected onto Principal Components', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_aspect('equal')

plt.tight_layout()
plt.savefig('docs/images/pca_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

# Variance explained
variance_explained = eigenvalues / np.sum(eigenvalues) * 100
print("Variance Explained:")
print(f"PC1: {variance_explained[0]:.2f}%")
print(f"PC2: {variance_explained[1]:.2f}%")
print(f"\nKey Concepts:")
print("- Covariance matrix")
print("- Eigenvalues and eigenvectors")
print("- Dimensionality reduction")
```

## Project 3: Linear Regression from Scratch

**Objective**: Implement linear regression using only matrix operations.

### Theory
Linear regression finds the best-fit line: `y = Xw + b`, where `w` are weights found using normal equation: `w = (X^T X)^(-1) X^T y`

### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """Fit linear regression using normal equation"""
        # Add bias term (column of ones)
        X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        
        # Normal equation: w = (X^T X)^(-1) X^T y
        XTX = X_with_bias.T @ X_with_bias
        XTy = X_with_bias.T @ y
        
        # Solve for weights
        params = np.linalg.solve(XTX, XTy)
        
        self.bias = params[0]
        self.weights = params[1:]
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        return X @ self.weights + self.bias
    
    def score(self, X, y):
        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X.flatten() + 1 + np.random.randn(100) * 2

# Fit model
model = LinearRegression()
model.fit(X, y)

# Make predictions
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = model.predict(X_test)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Regression line
axes[0].scatter(X, y, alpha=0.6, label='Data')
axes[0].plot(X_test, y_pred, 'r-', linewidth=2, label=f'Prediction: y = {model.weights[0]:.2f}x + {model.bias:.2f}')
axes[0].set_xlabel('X', fontsize=12)
axes[0].set_ylabel('y', fontsize=12)
axes[0].set_title('Linear Regression', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Residuals
y_pred_train = model.predict(X)
residuals = y - y_pred_train
axes[1].scatter(y_pred_train, residuals, alpha=0.6)
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Values', fontsize=12)
axes[1].set_ylabel('Residuals', fontsize=12)
axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('docs/images/linear_regression.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Model Parameters:")
print(f"Weights: {model.weights[0]:.4f}")
print(f"Bias: {model.bias:.4f}")
print(f"R² Score: {model.score(X, y):.4f}")
print(f"\nKey Concepts:")
print("- Matrix multiplication")
print("- Matrix inverse")
print("- Normal equation")
print("- Least squares")
```

## Project 4: Simple Neural Network

**Objective**: Build a simple neural network using only matrix operations.

### Theory
Neural networks are essentially matrix multiplications: `output = activation(input × weights + bias)`

### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights randomly
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def forward(self, X):
        """Forward propagation"""
        # Layer 1
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Layer 2 (output)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, output):
        """Backward propagation"""
        m = X.shape[0]
        
        # Output layer error
        dz2 = output - y
        dW2 = (1/m) * (self.a1.T @ dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer error
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.a1 * (1 - self.a1)
        dW1 = (1/m) * (X.T @ dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        return dW1, db1, dW2, db2
    
    def train(self, X, y, epochs=1000, learning_rate=0.1):
        """Train the network"""
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Calculate loss
            loss = np.mean((output - y) ** 2)
            losses.append(loss)
            
            # Backward pass
            dW1, db1, dW2, db2 = self.backward(X, y, output)
            
            # Update weights
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
        
        return losses

# XOR problem (non-linearly separable)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create and train network
nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
losses = nn.train(X, y, epochs=1000, learning_rate=0.5)

# Make predictions
predictions = nn.forward(X)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Training loss
axes[0].plot(losses, linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

# Predictions
axes[1].bar(range(4), y.flatten(), alpha=0.5, label='True', color='blue')
axes[1].bar(range(4), predictions.flatten(), alpha=0.7, label='Predicted', color='red')
axes[1].set_xlabel('Sample', fontsize=12)
axes[1].set_ylabel('Output', fontsize=12)
axes[1].set_title('XOR Problem: Predictions vs True Values', fontsize=14, fontweight='bold')
axes[1].set_xticks(range(4))
axes[1].set_xticklabels(['[0,0]', '[0,1]', '[1,0]', '[1,1]'])
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('docs/images/neural_network_xor.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPredictions:")
for i, (x, true_val, pred) in enumerate(zip(X, y, predictions)):
    print(f"Input: {x}, True: {true_val[0]}, Predicted: {pred[0]:.4f}")

print(f"\nKey Concepts:")
print("- Matrix multiplication for forward pass")
print("- Matrix transpose for backward pass")
print("- Gradient computation")
print("- Weight updates")
```

## Project 5: Data Preprocessing Pipeline

**Objective**: Implement data normalization and standardization using matrix operations.

### Theory
Data preprocessing uses linear algebra for:
- Mean centering: `X_centered = X - mean(X)`
- Standardization: `X_std = (X - mean) / std`
- Normalization: `X_norm = (X - min) / (max - min)`

### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

class DataPreprocessor:
    def __init__(self, method='standardize'):
        self.method = method
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
    
    def fit(self, X):
        """Calculate preprocessing parameters"""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        return self
    
    def transform(self, X):
        """Apply preprocessing"""
        if self.method == 'standardize':
            return (X - self.mean) / (self.std + 1e-8)
        elif self.method == 'normalize':
            return (X - self.min) / (self.max - self.min + 1e-8)
        elif self.method == 'center':
            return X - self.mean
        else:
            return X
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)

# Generate sample data
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                          n_informative=2, random_state=42)

# Apply different preprocessing methods
preprocessors = {
    'Original': None,
    'Centered': DataPreprocessor('center'),
    'Standardized': DataPreprocessor('standardize'),
    'Normalized': DataPreprocessor('normalize')
}

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, (name, preprocessor) in enumerate(preprocessors.items()):
    if preprocessor is None:
        X_processed = X
    else:
        X_processed = preprocessor.fit_transform(X)
    
    axes[idx].scatter(X_processed[:, 0], X_processed[:, 1], c=y, cmap='viridis', alpha=0.6)
    axes[idx].set_xlabel('Feature 1', fontsize=11)
    axes[idx].set_ylabel('Feature 2', fontsize=11)
    axes[idx].set_title(f'{name} Data', fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)
    
    # Add statistics
    if preprocessor is not None:
        stats_text = f"Mean: [{np.mean(X_processed[:, 0]):.2f}, {np.mean(X_processed[:, 1]):.2f}]\n"
        stats_text += f"Std: [{np.std(X_processed[:, 0]):.2f}, {np.std(X_processed[:, 1]):.2f}]"
        axes[idx].text(0.05, 0.95, stats_text, transform=axes[idx].transAxes,
                      fontsize=9, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
plt.savefig('docs/images/data_preprocessing.png', dpi=150, bbox_inches='tight')
plt.show()

print("Data Preprocessing Complete!")
print("Key Concepts:")
print("- Mean centering (subtraction)")
print("- Standardization (division)")
print("- Vector operations")
print("- Matrix broadcasting")
```

## Project 6: Face Recognition with Eigenfaces

**Objective**: Use PCA (eigenfaces) for face recognition.

### Theory
Eigenfaces are the principal components of face images. We project faces onto these components for recognition.

### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

# Load face dataset (or create synthetic faces)
try:
    faces = fetch_olivetti_faces()
    face_images = faces.images
    face_data = faces.data
    print(f"Loaded {len(face_images)} face images")
except:
    # Create synthetic faces if dataset unavailable
    print("Creating synthetic face data...")
    np.random.seed(42)
    face_images = []
    for i in range(40):
        # Create a simple face pattern
        face = np.random.rand(64, 64)
        face[20:44, 20:44] = 0.8  # Face region
        face[25:30, 28:36] = 0.3  # Eyes
        face[35:40, 30:34] = 0.3  # Nose
        face[42:45, 28:36] = 0.3  # Mouth
        face_images.append(face)
    face_data = np.array([img.flatten() for img in face_images])

# Reshape to 2D
X = face_data

# Center the data
X_centered = X - np.mean(X, axis=0)

# Calculate covariance matrix
cov_matrix = np.cov(X_centered.T)

# Find top k eigenvectors (eigenfaces)
k = 16  # Number of eigenfaces to use
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
idx = eigenvalues.argsort()[::-1]
eigenfaces = eigenvectors[:, idx[:k]].real

# Project faces onto eigenfaces
face_projections = X_centered @ eigenfaces

# Reconstruct faces using eigenfaces
face_reconstructed = face_projections @ eigenfaces.T + np.mean(X, axis=0)

# Visualize
fig, axes = plt.subplots(4, 4, figsize=(12, 12))

# Show first 16 eigenfaces
for i in range(16):
    row, col = i // 4, i % 4
    eigenface = eigenfaces[:, i].reshape(64, 64)
    axes[row, col].imshow(eigenface, cmap='gray')
    axes[row, col].set_title(f'Eigenface {i+1}', fontsize=9, fontweight='bold')
    axes[row, col].axis('off')

plt.suptitle('Eigenfaces (Principal Components)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('docs/images/eigenfaces.png', dpi=150, bbox_inches='tight')
plt.show()

# Compare original and reconstructed
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i in range(5):
    # Original
    axes[0, i].imshow(face_images[i], cmap='gray')
    axes[0, i].set_title(f'Original Face {i+1}', fontsize=10, fontweight='bold')
    axes[0, i].axis('off')
    
    # Reconstructed
    reconstructed = face_reconstructed[i].reshape(64, 64)
    axes[1, i].imshow(reconstructed, cmap='gray')
    axes[1, i].set_title(f'Reconstructed (k={k})', fontsize=10, fontweight='bold')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('docs/images/face_reconstruction.png', dpi=150, bbox_inches='tight')
plt.show()

# Calculate reconstruction error
reconstruction_error = np.mean((X - face_reconstructed) ** 2)
print(f"Reconstruction Error (MSE): {reconstruction_error:.4f}")
print(f"Variance Explained: {np.sum(eigenvalues[:k]) / np.sum(eigenvalues) * 100:.2f}%")
print(f"\nKey Concepts:")
print("- PCA for dimensionality reduction")
print("- Eigenvectors as basis")
print("- Matrix projection")
print("- Image compression")
```

## Summary

These projects demonstrate practical applications of linear algebra:

1. **Image Transformations** - Matrix operations for geometric transformations
2. **PCA** - Eigenvalues/eigenvectors for dimensionality reduction
3. **Linear Regression** - Matrix operations for solving least squares
4. **Neural Networks** - Matrix multiplication for forward/backward propagation
5. **Data Preprocessing** - Vector operations for normalization
6. **Eigenfaces** - PCA for face recognition and compression

Each project includes:
- Theory explanation
- Complete implementation
- Visualizations
- Key concepts learned

Run each project to see linear algebra in action!
