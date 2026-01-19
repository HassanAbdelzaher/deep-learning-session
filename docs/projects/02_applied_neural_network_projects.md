# Applied Neural Network Projects

## Table of Contents
1. [Project 1: Handwritten Digit Recognition (MNIST)](#project-1-handwritten-digit-recognition-mnist)
2. [Project 2: Binary Classification - Spam Detection](#project-2-binary-classification---spam-detection)
3. [Project 3: Multi-class Classification - Iris Dataset](#project-3-multi-class-classification---iris-dataset)
4. [Project 4: Regression - House Price Prediction](#project-4-regression---house-price-prediction)
5. [Project 5: XOR Problem - Non-linearity Demonstration](#project-5-xor-problem---non-linearity-demonstration)
6. [Project 6: Image Classification with CNN](#project-6-image-classification-with-cnn)
7. [Project 7: Student Degree Classification](#project-7-student-degree-classification)
8. [Project 8: Student Degree Classification (Advanced - High Accuracy)](#project-8-student-degree-classification-advanced---high-accuracy)

## Project 1: Handwritten Digit Recognition (MNIST)

**Objective**: Build a neural network to classify handwritten digits (0-9) from the MNIST dataset.

### Theory
MNIST is a classic dataset of 28×28 grayscale images of handwritten digits. We'll use a feedforward neural network with:
- Input layer: 784 neurons (28×28 flattened)
- Hidden layers: Multiple fully connected layers
- Output layer: 10 neurons (one for each digit)

### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.deep_learning.neural_networks import SimpleNeuralNetwork

# Load MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist.data, mnist.target.astype(int)

# Normalize pixel values to [0, 1]
X = X / 255.0

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# One-hot encode labels
def one_hot_encode(y, num_classes=10):
    encoded = np.zeros((len(y), num_classes))
    encoded[np.arange(len(y)), y] = 1
    return encoded

y_train_encoded = one_hot_encode(y_train)
y_test_encoded = one_hot_encode(y_test)

# Create and train neural network
print("Creating neural network...")
nn = SimpleNeuralNetwork(layers=[784, 128, 64, 10], learning_rate=0.01)

print("Training neural network...")
loss_history = nn.train(X_train, y_train_encoded, epochs=50, verbose=True)

# Evaluate
predictions = nn.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
accuracy = np.mean(predicted_classes == y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Visualize results
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i in range(10):
    idx = np.where(y_test == i)[0][0]
    axes[i//5, i%5].imshow(X_test[idx].reshape(28, 28), cmap='gray')
    axes[i//5, i%5].set_title(f'True: {y_test[idx]}\nPred: {predicted_classes[idx]}')
    axes[i//5, i%5].axis('off')
plt.suptitle('Sample Predictions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('docs/images/mnist_predictions.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss - MNIST Classification')
plt.grid(True, alpha=0.3)
plt.savefig('docs/images/mnist_training_loss.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Learning Outcomes
- Understand how to preprocess image data for neural networks
- Learn to handle multi-class classification
- Practice with real-world dataset
- Visualize model predictions

---

## Project 2: Binary Classification - Spam Detection

**Objective**: Build a neural network to classify emails as spam or not spam.

### Theory
Binary classification requires:
- Input features: Email characteristics (word frequencies, length, etc.)
- Output: Single neuron with sigmoid activation (probability of spam)
- Loss function: Binary cross-entropy

### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.deep_learning.neural_networks import SimpleNeuralNetwork

# Generate synthetic email dataset
# In practice, you'd extract features from real emails
print("Generating synthetic email dataset...")
X, y = make_classification(
    n_samples=5000,
    n_features=20,  # Email features (word counts, length, etc.)
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

# Reshape y for binary classification
y = y.reshape(-1, 1)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train network
print("Creating neural network...")
nn = SimpleNeuralNetwork(layers=[20, 32, 16, 1], learning_rate=0.01)

print("Training neural network...")
loss_history = nn.train(X_train, y_train, epochs=100, verbose=True)

# Evaluate
predictions = nn.predict(X_test)
predicted_classes = (predictions > 0.5).astype(int)
accuracy = np.mean(predicted_classes == y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, predicted_classes)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, predicted_classes))

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(loss_history)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].grid(True, alpha=0.3)

axes[1].imshow(cm, cmap='Blues', interpolation='nearest')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title('Confusion Matrix')
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14)
plt.tight_layout()
plt.savefig('docs/images/spam_detection_results.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Learning Outcomes
- Binary classification with neural networks
- Feature scaling importance
- Confusion matrix interpretation
- Real-world application

---

## Project 3: Multi-class Classification - Iris Dataset

**Objective**: Classify iris flowers into three species using neural networks.

### Theory
The Iris dataset has 4 features (sepal length, sepal width, petal length, petal width) and 3 classes (Setosa, Versicolor, Virginica).

### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.deep_learning.neural_networks import SimpleNeuralNetwork

# Load Iris dataset
print("Loading Iris dataset...")
iris = load_iris()
X, y = iris.data, iris.target

# One-hot encode
def one_hot_encode(y, num_classes=3):
    encoded = np.zeros((len(y), num_classes))
    encoded[np.arange(len(y)), y] = 1
    return encoded

y_encoded = one_hot_encode(y)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train network
print("Creating neural network...")
nn = SimpleNeuralNetwork(layers=[4, 8, 3], learning_rate=0.01)

print("Training neural network...")
loss_history = nn.train(X_train, y_train, epochs=200, verbose=True)

# Evaluate
predictions = nn.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = np.argmax(y_test, axis=1)
accuracy = np.mean(predicted_classes == actual_classes)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Visualize decision boundaries
from matplotlib.colors import ListedColormap
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
feature_pairs = [(0, 1), (0, 2), (0, 3), (2, 3)]
feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

for idx, (i, j) in enumerate(feature_pairs):
    ax = axes[idx//2, idx%2]
    
    # Create mesh
    h = 0.02
    x_min, x_max = X[:, i].min() - 1, X[:, i].max() + 1
    y_min, y_max = X[:, j].min() - 1, X[:, j].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Predict on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    # Use mean values for other features
    full_points = np.zeros((len(mesh_points), 4))
    full_points[:, i] = mesh_points[:, 0]
    full_points[:, j] = mesh_points[:, 1]
    full_points[:, [k for k in range(4) if k not in [i, j]]] = X[:, [k for k in range(4) if k not in [i, j]]].mean(axis=0)
    
    full_points_scaled = scaler.transform(full_points)
    Z = nn.predict(full_points_scaled)
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    
    # Plot
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']))
    scatter = ax.scatter(X[:, i], X[:, j], c=y, cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF']))
    ax.set_xlabel(feature_names[i])
    ax.set_ylabel(feature_names[j])
    ax.set_title(f'{feature_names[i]} vs {feature_names[j]}')

plt.suptitle('Decision Boundaries - Iris Classification', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('docs/images/iris_decision_boundaries.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Learning Outcomes
- Multi-class classification
- Feature visualization
- Decision boundary understanding
- Small dataset handling

---

## Project 4: Regression - House Price Prediction

**Objective**: Predict house prices using neural networks for regression.

### Theory
Regression with neural networks:
- Output layer: Single neuron (no activation or linear activation)
- Loss function: Mean Squared Error (MSE)
- Features: House characteristics (size, bedrooms, location, etc.)

### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from src.deep_learning.neural_networks import SimpleNeuralNetwork

# Generate synthetic house price dataset
print("Generating synthetic house price dataset...")
X, y = make_regression(
    n_samples=1000,
    n_features=10,  # Features: size, bedrooms, bathrooms, age, location, etc.
    n_informative=8,
    noise=20,
    random_state=42
)

# Reshape y
y = y.reshape(-1, 1)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Create and train network
print("Creating neural network...")
nn = SimpleNeuralNetwork(layers=[10, 32, 16, 1], learning_rate=0.01)

print("Training neural network...")
loss_history = nn.train(X_train, y_train, epochs=200, verbose=True)

# Evaluate
predictions = nn.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"\nTest MSE: {mse:.4f}")
print(f"Test R²: {r2:.4f}")

# Inverse transform for visualization
y_test_original = scaler_y.inverse_transform(y_test)
predictions_original = scaler_y.inverse_transform(predictions)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(loss_history)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].set_title('Training Loss')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_test_original, predictions_original, alpha=0.6)
axes[1].plot([y_test_original.min(), y_test_original.max()],
            [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Price')
axes[1].set_ylabel('Predicted Price')
axes[1].set_title(f'Predictions vs Actual (R² = {r2:.3f})')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('docs/images/house_price_prediction.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Learning Outcomes
- Regression with neural networks
- Feature scaling for regression
- Evaluation metrics (MSE, R²)
- Real-world prediction problems

---

## Project 5: XOR Problem - Non-linearity Demonstration

**Objective**: Solve the XOR problem to demonstrate the importance of non-linear activation functions.

### Theory
XOR is not linearly separable. A single-layer perceptron cannot solve it, but a multi-layer network with non-linear activations can.

### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from src.deep_learning.neural_networks import SimpleNeuralNetwork, visualize_perceptron_decision_boundary

# XOR problem data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

print("XOR Problem:")
print("Input | Output")
print("------|-------")
for i in range(4):
    print(f"{X[i]} | {y[i][0]}")

# Create and train network
print("\nCreating neural network...")
nn = SimpleNeuralNetwork(layers=[2, 4, 1], learning_rate=0.5)

print("Training neural network...")
loss_history = nn.train(X, y, epochs=2000, verbose=True)

# Evaluate
predictions = nn.predict(X)
print("\nPredictions:")
print("Input | Target | Prediction")
print("------|--------|-----------")
for i in range(4):
    print(f"{X[i]} | {y[i][0]} | {predictions[i][0]:.4f}")

# Visualize decision boundary
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot data points
axes[0].scatter(X[y.flatten()==0, 0], X[y.flatten()==0, 1], 
               c='blue', s=200, marker='o', label='Class 0', edgecolors='black')
axes[0].scatter(X[y.flatten()==1, 0], X[y.flatten()==1, 1], 
               c='red', s=200, marker='s', label='Class 1', edgecolors='black')

# Create decision boundary
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = nn.predict(grid_points)
Z = Z.reshape(xx.shape)

axes[0].contourf(xx, yy, Z, levels=50, alpha=0.3, cmap='RdYlBu')
axes[0].set_xlabel('x₁')
axes[0].set_ylabel('x₂')
axes[0].set_title('XOR Decision Boundary')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(-0.5, 1.5)
axes[0].set_ylim(-0.5, 1.5)

# Plot training loss
axes[1].plot(loss_history)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Training Loss')
axes[1].set_yscale('log')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('docs/images/xor_problem.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nKey Insight: XOR requires non-linear activation functions!")
print("A single-layer perceptron cannot solve XOR, but a multi-layer network can.")
```

### Learning Outcomes
- Importance of non-linearity
- Multi-layer networks necessity
- Decision boundary visualization
- Classic problem in neural networks

---

## Project 6: Image Classification with CNN

**Objective**: Build a Convolutional Neural Network for image classification.

### Theory
CNNs are specialized for image data:
- Convolutional layers detect local patterns
- Pooling layers reduce dimensionality
- Fully connected layers make final predictions

### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.deep_learning.cnn import SimpleCNN, visualize_cnn_architecture

# Generate synthetic image data (simulating CIFAR-10 style)
print("Generating synthetic image dataset...")
np.random.seed(42)
num_samples = 2000
num_classes = 10

# Create synthetic 32x32 RGB images
X = np.random.rand(num_samples, 3, 32, 32).astype(np.float32)
y = np.random.randint(0, num_classes, num_samples)

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y).long()

# Create dataset and dataloader
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model
print("Creating CNN...")
model = SimpleCNN(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
train_losses = []
test_accuracies = []

print("Training CNN...")
for epoch in range(num_epochs):
    # Training
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    train_losses.append(epoch_loss / len(train_loader))
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

# Visualize results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(train_losses)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].grid(True, alpha=0.3)

axes[1].plot(test_accuracies)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Test Accuracy')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('docs/images/cnn_training_results.png', dpi=150, bbox_inches='tight')
plt.show()

# Visualize architecture
visualize_cnn_architecture(save_path='docs/images/cnn_architecture.png')
```

### Learning Outcomes
- CNN architecture understanding
- Image data preprocessing
- Convolution and pooling operations
- Transfer learning concepts

---

## Project 7: Student Degree Classification

**Objective**: Build a neural network to classify students into degree categories based on their academic performance using a CSV dataset.

### Theory

This project demonstrates:
- **CSV Data Processing**: Loading and preprocessing CSV files
- **Multi-class Classification**: 5 degree categories (Bad, Acceptable, Good, Very Good, Excellent)
- **Feature Engineering**: Using multiple academic features to predict performance
- **Real-world Application**: Practical use case for educational data

**Degree Categories** (based on final score):
- **Bad**: score < 50
- **Acceptable**: 50 ≤ score < 65
- **Good**: 65 ≤ score < 75
- **Very Good**: 75 ≤ score < 85
- **Excellent**: score ≥ 85

### Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'src'))
from deep_learning.neural_networks import SimpleNeuralNetwork

# Load dataset from CSV
print("Loading student dataset...")
df = pd.read_csv('data/neural_networks/student_degree_dataset.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Prepare features and labels
feature_columns = ['attendance', 'quiz_avg', 'assignment_avg', 'midterm_score',
                   'project_score', 'study_hours_per_week', 'participation_score']
X = df[feature_columns].values
y_categories = df['degree_category'].values

# Map categories to numbers
category_mapping = {
    'Bad': 0,
    'Acceptable': 1,
    'Good': 2,
    'Very Good': 3,
    'Excellent': 4
}
y = np.array([category_mapping[cat] for cat in y_categories])

# One-hot encode
def one_hot_encode(y, num_classes=5):
    encoded = np.zeros((len(y), num_classes))
    encoded[np.arange(len(y)), y] = 1
    return encoded

y_encoded = one_hot_encode(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Features: {X_train.shape[1]}")
print(f"Classes: {y_encoded.shape[1]}")

# Create and train neural network
print("\nCreating neural network...")
nn = SimpleNeuralNetwork(layers=[7, 32, 16, 5], learning_rate=0.01)

print("Training neural network...")
loss_history = nn.train(X_train, y_train, epochs=200, verbose=True)

# Evaluate
predictions = nn.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = np.argmax(y_test, axis=1)
accuracy = np.mean(predicted_classes == actual_classes)

print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Classification report
category_names = ['Bad', 'Acceptable', 'Good', 'Very Good', 'Excellent']
print("\nClassification Report:")
print(classification_report(actual_classes, predicted_classes, target_names=category_names))

# Confusion matrix
cm = confusion_matrix(actual_classes, predicted_classes)

# Visualize results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Training loss
axes[0].plot(loss_history)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].grid(True, alpha=0.3)

# Confusion matrix
im = axes[1].imshow(cm, cmap='Blues', interpolation='nearest')
axes[1].set_xticks(range(5))
axes[1].set_yticks(range(5))
axes[1].set_xticklabels(category_names, rotation=45, ha='right')
axes[1].set_yticklabels(category_names)
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title('Confusion Matrix')
for i in range(5):
    for j in range(5):
        axes[1].text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=axes[1])
plt.tight_layout()
plt.savefig('docs/images/student_degree_classification.png', dpi=150, bbox_inches='tight')
plt.show()

# Feature importance analysis
print("\nFeature Statistics by Degree Category:")
for category in category_names:
    category_idx = category_mapping[category]
    category_data = df[df['degree_category'] == category]
    print(f"\n{category}:")
    for col in feature_columns:
        print(f"  {col}: {category_data[col].mean():.2f} (std: {category_data[col].std():.2f})")
```

### Learning Outcomes
- CSV data loading and preprocessing
- Multi-class classification with 5 classes
- Feature scaling importance
- Real-world educational data application
- Confusion matrix interpretation
- Feature analysis by category

---

## Project 8: Student Degree Classification (Advanced - High Accuracy)

**Objective**: Build an advanced neural network to achieve high accuracy (>85%) in student degree classification using improved techniques and optimizations.

### Theory

This advanced project builds upon Project 7 with significant improvements:

- **Deeper Architecture**: 5-layer network (Poly→64→128→64→32→5) vs 3-layer (7→32→16→5)
- **Feature Engineering**: Polynomial features (degree 2, interaction only) expand from 7 to 28 features
- **Class Balancing**: Oversampling minority classes to handle imbalanced data
- **Learning Rate Scheduling**: Adaptive learning rate with exponential decay (0.01 → 0.0001)
- **Early Stopping**: Prevents overfitting with validation monitoring
- **Extended Training**: 500 epochs with patience-based early stopping
- **Comprehensive Evaluation**: Multiple metrics (accuracy, F1, precision, recall)

**Key Improvements Over Project 7**:
1. Polynomial features capture feature interactions
2. Class balancing improves minority class performance
3. Learning rate scheduling enables fine-tuning
4. Early stopping prevents overfitting
5. Deeper network learns more complex patterns

### Implementation

See the complete implementation in `notebooks/10_project_8_student_degree_classification_advanced.ipynb`.

**Key Components**:

```python
# 1. Feature Engineering with Polynomial Features
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_poly = poly.fit_transform(X)  # 7 → 28 features

# 2. Class Balancing with Oversampling
def oversample_minority_classes(X, y, y_labels, target_samples=300):
    # Oversample minority classes to balance dataset
    ...

# 3. Advanced Neural Network with Learning Rate Scheduling
class AdvancedNeuralNetwork(SimpleNeuralNetwork):
    def __init__(self, layers, learning_rate=0.01, lr_decay=0.98, min_lr=0.0001):
        super().__init__(layers, learning_rate)
        self.lr_decay = lr_decay
        self.min_lr = min_lr
    
    def train_with_early_stopping(self, X_train, y_train, X_val, y_val, epochs=500):
        # Training with learning rate decay and early stopping
        ...

# 4. Deeper Architecture
nn_advanced = AdvancedNeuralNetwork(
    layers=[28, 64, 128, 64, 32, 5],  # Polynomial features → deeper network
    learning_rate=0.01,
    lr_decay=0.98,
    min_lr=0.0001
)

# 5. Comprehensive Evaluation
from sklearn.metrics import (accuracy_score, f1_score, 
                           precision_score, recall_score)

accuracy = accuracy_score(actual_classes, predicted_classes)
f1_macro = f1_score(actual_classes, predicted_classes, average='macro')
f1_weighted = f1_score(actual_classes, predicted_classes, average='weighted')
```

### Performance Comparison

| Metric | Project 7 | Project 8 | Improvement |
|--------|-----------|-----------|-------------|
| Architecture | 7→32→16→5 | Poly→64→128→64→32→5 | Deeper |
| Features | 7 (original) | 28 (polynomial) | 4x more |
| Class Balancing | No | Yes (oversampling) | Balanced |
| Learning Rate | Fixed 0.01 | Adaptive (0.01→0.0001) | Scheduled |
| Epochs | 200 | 500 (early stop) | Extended |
| **Accuracy** | **~62%** | **>85%** | **+23%** |

### Key Techniques Demonstrated

1. **Polynomial Feature Engineering**
   - Captures feature interactions
   - Expands feature space for better learning
   - Interaction-only reduces dimensionality

2. **Class Balancing**
   - Oversamples minority classes
   - Improves model performance on rare classes
   - Handles imbalanced datasets

3. **Learning Rate Scheduling**
   - Exponential decay: `lr = initial_lr * (decay ^ epoch)`
   - Enables fine-tuning in later epochs
   - Prevents overshooting optimal solution

4. **Early Stopping**
   - Monitors validation loss
   - Stops training when no improvement
   - Prevents overfitting

5. **Deeper Architecture**
   - More layers for complex pattern learning
   - Better feature extraction
   - Improved generalization

### Learning Outcomes

- Advanced feature engineering techniques
- Handling imbalanced datasets
- Learning rate scheduling strategies
- Early stopping for regularization
- Comprehensive model evaluation
- Performance optimization techniques
- Comparison of different approaches
- Feature importance analysis

### Expected Results

- **Accuracy**: >85% (vs ~62% in Project 7)
- **F1 Score**: >0.85 (macro and weighted)
- **Balanced Performance**: Good across all classes
- **Robust Model**: Generalizes well to new data

---

## Summary

These projects cover:
1. **Image Classification** - MNIST digit recognition
2. **Binary Classification** - Spam detection
3. **Multi-class Classification** - Iris species
4. **Regression** - House price prediction
5. **Non-linearity** - XOR problem
6. **CNN Applications** - Image classification
7. **CSV Data Processing** - Student degree classification
8. **Advanced Techniques** - High-accuracy classification with feature engineering and optimization

Each project demonstrates different aspects of neural networks and provides hands-on experience with real-world applications.

## Next Steps

1. Try modifying hyperparameters (learning rate, layers, neurons)
2. Experiment with different activation functions
3. Add regularization (dropout, L2)
4. Implement early stopping
5. Try different optimizers (Adam, RMSprop)
6. Apply to your own datasets!
