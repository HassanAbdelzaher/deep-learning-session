# Neural Networks Fundamentals

## Table of Contents
1. [Introduction](#introduction)
2. [Perceptron](#perceptron)
3. [Multi-Layer Perceptron](#multi-layer-perceptron)
4. [Forward Propagation](#forward-propagation)
5. [Backpropagation](#backpropagation)
6. [Training Process](#training-process)
7. [Activation Functions](#activation-functions)

## Introduction

Neural networks are computational models inspired by biological neurons. They consist of:
- **Input layer**: Receives data
- **Hidden layers**: Process information
- **Output layer**: Produces predictions
- **Weights and biases**: Learnable parameters
- **Activation functions**: Introduce non-linearity

## Perceptron

### What is a Perceptron?

A perceptron is the simplest neural network - a single neuron that can perform binary classification. It takes multiple inputs, applies weights and a bias, and produces a binary output (0 or 1) using a step activation function.

**Mathematical Formula:**
```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
y = step(z) = {1 if z ≥ 0, 0 otherwise}
```

### Perceptron Learning Algorithm

The perceptron learning algorithm updates weights and bias when it makes a mistake:

```
If prediction is wrong:
    w = w + learning_rate × (true_label - prediction) × input
    b = b + learning_rate × (true_label - prediction)
```

### Practical Implementations

We provide two complete perceptron implementations:

#### 1. Two-Feature Perceptron (`perceptron.py`)

A perceptron that uses **Exam Score** and **Attendance Percentage** to predict whether a student will pass or fail.

**Features:**
- 2 input features: Exam Score (40-100), Attendance (40-100%)
- Binary classification: Pass (1) or Fail (0)
- Rule: Pass if exam score ≥ 60 OR attendance ≥ 60
- Visualizes 2D decision boundary

**Usage:**
```python
# Run the complete perceptron example
python src/deep_learning/perceptron.py
```

**Key Functions:**
- `generate_data(num_samples, seed)`: Generates synthetic student data
- `step(z)`: Step activation function
- `predict_one(x, w, b)`: Predict for single sample
- `predict(X, w, b)`: Predict for multiple samples
- `train_perceptron(X, y, lr, epochs)`: Train the perceptron
- `plot_decision_boundary(X, y, w, b)`: Visualize data and decision boundary

**Example Output:**
- Trained weights and bias
- Decision boundary visualization (2D plot)
- Training error evolution
- Predictions on new students

#### 2. One-Feature Perceptron (`perceptron2.py`)

A simpler perceptron that uses only **Exam Score** to predict pass/fail.

**Features:**
- 1 input feature: Exam Score (40-100)
- Binary classification: Pass (1) or Fail (0)
- Rule: Pass if exam score ≥ 60
- Visualizes 1D decision boundary (vertical line)

**Usage:**
```python
# Run the simple perceptron example
python src/deep_learning/perceptron2.py
```

**Key Differences from 2-feature version:**
- Simpler decision boundary (single threshold)
- Faster training
- Easier to understand for beginners

### Running the Perceptron Examples

Both implementations include:
1. **Data Generation**: Synthetic student records
2. **Training**: Perceptron learning algorithm with early stopping
3. **Visualization**: Decision boundary plots
4. **Error Tracking**: Training error evolution over epochs
5. **Prediction**: Test on new student data
6. **Interactive Testing**: User input for custom predictions

**Example: Training a Perceptron**

```python
# Direct execution (recommended)
python src/deep_learning/perceptron.py

# Or import functions (note: avoid importing from __init__.py due to torch dependency)
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))

# For perceptron.py (2 features)
exec(open('src/deep_learning/perceptron.py').read())

# Or use Makefile commands
# make run-perceptron        # 2-feature version
# make run-perceptron-simple # 1-feature version
```

**Complete Workflow:**

1. **Data Generation**: Creates synthetic student records
2. **Training**: Learns weights and bias using perceptron algorithm
3. **Visualization**: Shows decision boundary and training progress
4. **Prediction**: Tests on new student data
5. **Interactive Testing**: Allows user input for custom predictions

### Perceptron Limitations

1. **Linearly Separable Data Only**: Can only learn linearly separable patterns
2. **Binary Classification**: Only outputs 0 or 1
3. **No Probabilities**: Doesn't provide confidence scores
4. **Cannot Solve XOR**: Requires non-linear decision boundaries
5. **Not Differentiable**: Step function is not differentiable → can't use gradient descent properly

**Why Multi-Layer Networks?**
- Single perceptrons can't learn complex patterns
- Multiple layers with non-linear activations enable complex decision boundaries
- This is why we need Multi-Layer Perceptrons (MLPs) for real-world problems

## Logistic Regression: The Bridge to Neural Networks

### Why Logistic Regression?

**Perceptron Problem:**
- Uses step function (hard 0/1) → **Not differentiable** → Can't use calculus properly

**Logistic Regression Solution:**
- Uses sigmoid function (smooth 0-1) → **Differentiable** → Enables Gradient Descent!

### Key Differences from Perceptron

| Feature | Perceptron | Logistic Regression |
|---------|-----------|---------------------|
| Activation | Step function | Sigmoid function |
| Output | Hard 0/1 | Probability (0-1) |
| Training | Perceptron update rule | Gradient Descent |
| Loss Function | Error count | Binary Cross-Entropy |
| Differentiable | ❌ No | ✅ Yes |

### Mathematical Foundation

**Core Formula:**
```
ŷ = σ(w·x + b)
σ(z) = 1 / (1 + e^(-z))
```

Where:
- `σ` is the sigmoid function (smooth, differentiable)
- `w` are weights
- `b` is bias
- Output is a **probability** between 0 and 1

### Sigmoid Activation Function

The sigmoid function is the key innovation:

```python
def sigmoid(z):
    """Sigmoid activation: smooth, differentiable, outputs probabilities"""
    return 1 / (1 + np.exp(-z))
```

**Properties:**
- ✅ Smooth and differentiable everywhere
- ✅ Output range: (0, 1) → probabilities!
- ✅ S-shaped curve
- ✅ Enables gradient descent

### Binary Cross-Entropy Loss

Logistic regression uses a proper loss function:

```
L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

**Why this loss?**
- Penalizes confident wrong predictions heavily
- Works perfectly with probabilities
- Differentiable everywhere

### Gradient Descent Training

Unlike perceptron's update rule, logistic regression uses gradient descent:

```python
# Compute gradients using calculus
dw, db = compute_gradients(X, y, w, b)

# Update using gradient descent
w = w - learning_rate * dw
b = b - learning_rate * db
```

**The Power of Calculus:**
- We can compute exact gradients
- We can optimize using gradient descent
- This is the foundation of all neural network training!

### Practical Implementation

We provide a complete logistic regression implementation (`logistic_regression.py`):

**Features:**
- 2 input features: Exam Score, Attendance Percentage
- Binary classification: Pass (1) or Fail (0)
- Sigmoid activation function
- Binary Cross-Entropy loss
- Gradient Descent training
- Probability outputs (not just 0/1)
- Visualizations: decision boundary, probability surface, loss curve

**Usage:**
```python
# Run the complete logistic regression example
python src/deep_learning/logistic_regression.py

# Or use Makefile
make run-logistic-regression
```

**What You'll See:**
1. **Training Progress**: Loss decreasing over epochs
2. **Decision Boundary**: Smooth probability surface (not a hard line!)
3. **3D Probability Surface**: See how probabilities change across feature space
4. **Step vs Sigmoid Comparison**: Visual difference between perceptron and logistic regression
5. **Predictions with Probabilities**: Get confidence scores, not just 0/1

**Example Output:**
```
Student 1: Exam Score: 72.0, Attendance: 68.0%
  Probability of Pass: 0.8542 (85.42%)
  Prediction: Pass ✅
  Confidence: High
```

### Why This Matters

**Logistic Regression is the bridge from Perceptron to Neural Networks because:**

1. ✅ **Differentiable**: Enables gradient descent (foundation of all deep learning)
2. ✅ **Probabilities**: Provides confidence scores, not just decisions
3. ✅ **Proper Loss Function**: Uses mathematical loss functions
4. ✅ **Gradient Computation**: Shows how to compute gradients using calculus
5. ✅ **Foundation for Neural Networks**: All neural networks use these concepts!

**Next Step:** Multi-Layer Perceptrons use the same principles (sigmoid, gradient descent, loss functions) but with multiple layers!

## Multi-Layer Perceptron

### Network Architecture

```python
def visualize_mlp():
    """Visualize a multi-layer perceptron"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Layer positions
    layers = [
        [(0.5, i) for i in [4, 3, 2, 1]],  # Input layer
        [(2.5, i) for i in [4.5, 3.5, 2.5, 1.5]],  # Hidden layer 1
        [(4.5, i) for i in [4, 3, 2]],  # Hidden layer 2
        [(6.5, i) for i in [3.5, 2.5]]  # Output layer
    ]
    
    layer_labels = [
        ['x₁', 'x₂', 'x₃', 'x₄'],
        ['h₁₁', 'h₁₂', 'h₁₃', 'h₁₄'],
        ['h₂₁', 'h₂₂', 'h₂₃'],
        ['ŷ₁', 'ŷ₂']
    ]
    
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    
    # Draw nodes and connections
    for layer_idx, (positions, labels, color) in enumerate(zip(layers, layer_labels, colors)):
        for pos, label in zip(positions, labels):
            circle = Circle(pos, 0.25, color=color, ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], label, ha='center', va='center',
                   fontsize=10, fontweight='bold')
        
        # Draw connections to next layer
        if layer_idx < len(layers) - 1:
            for pos1 in positions:
                for pos2 in layers[layer_idx + 1]:
                    ax.plot([pos1[0]+0.25, pos2[0]-0.25], [pos1[1], pos2[1]],
                           'gray', linewidth=0.5, alpha=0.3)
    
    # Layer labels
    layer_names = ['Input\nLayer', 'Hidden\nLayer 1', 'Hidden\nLayer 2', 'Output\nLayer']
    for i, (name, x_pos) in enumerate(zip(layer_names, [0.5, 2.5, 4.5, 6.5])):
        ax.text(x_pos, 5.5, name, ha='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(0, 6)
    ax.set_title('Multi-Layer Perceptron (MLP)', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('docs/images/mlp_architecture.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_mlp()
```

## Forward Propagation

Forward propagation computes predictions by passing data through the network.

```python
def forward_propagation_example():
    """Demonstrate forward propagation step by step"""
    # Simple network: 2 inputs -> 2 hidden -> 1 output
    np.random.seed(42)
    
    # Weights and biases
    W1 = np.array([[0.5, -0.3], [0.2, 0.8]])  # 2x2
    b1 = np.array([0.1, -0.2])
    W2 = np.array([[0.7, -0.5]])  # 1x2
    b2 = np.array([0.3])
    
    # Input
    X = np.array([[1.0, 2.0]])
    
    print("Forward Propagation Example:")
    print(f"Input: X = {X}")
    
    # Layer 1
    z1 = np.dot(X, W1) + b1
    print(f"\nLayer 1 (before activation):")
    print(f"  z1 = X × W1 + b1 = {z1}")
    
    # Activation (ReLU)
    a1 = np.maximum(0, z1)
    print(f"  a1 = ReLU(z1) = {a1}")
    
    # Layer 2 (output)
    z2 = np.dot(a1, W2.T) + b2
    print(f"\nLayer 2 (output):")
    print(f"  z2 = a1 × W2 + b2 = {z2}")
    
    # Final output (sigmoid for binary classification)
    output = 1 / (1 + np.exp(-z2))
    print(f"  y = sigmoid(z2) = {output}")
    
    return output

forward_propagation_example()
```

## Backpropagation

Backpropagation computes gradients by propagating errors backward through the network.

```python
def backpropagation_visualization():
    """Visualize the backpropagation process"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Forward pass (blue)
    forward_nodes = {
        'x': (1, 4),
        'h1': (3, 4),
        'h2': (3, 2),
        'y': (5, 3),
        'L': (7, 3)
    }
    
    # Draw forward pass
    for name, (x, y) in forward_nodes.items():
        if name == 'L':
            rect = mpatches.Rectangle((x-0.3, y-0.2), 0.6, 0.4, 
                                     color='lightcoral', ec='black', linewidth=2)
            ax.add_patch(rect)
        else:
            circle = Circle((x, y), 0.25, color='lightblue', ec='black', linewidth=2)
            ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Forward arrows
    forward_arrows = [
        ((1.25, 4), (2.75, 4), 'Forward'),
        ((3.25, 4), (4.75, 3), 'Forward'),
        ((3.25, 2), (4.75, 3), 'Forward'),
        ((5.25, 3), (6.7, 3), 'Forward')
    ]
    
    for (x1, y1), (x2, y2), label in forward_arrows:
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', 
                               mutation_scale=20, linewidth=2, color='blue', alpha=0.7)
        ax.add_patch(arrow)
    
    # Backward pass (red, dashed)
    backward_arrows = [
        ((6.7, 3), (5.25, 3), '∂L/∂y'),
        ((4.75, 3), (3.25, 4), '∂L/∂h1'),
        ((4.75, 3), (3.25, 2), '∂L/∂h2'),
        ((2.75, 4), (1.25, 4), '∂L/∂x')
    ]
    
    for (x1, y1), (x2, y2), label in backward_arrows:
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', 
                               mutation_scale=20, linewidth=2, color='red', 
                               linestyle='--', alpha=0.7)
        ax.add_patch(arrow)
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.15, label, fontsize=9, ha='center', color='red',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.text(4, 0.5, 'Blue: Forward Pass (Compute Output)\nRed: Backward Pass (Compute Gradients)',
           ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 5)
    ax.set_title('Forward and Backward Propagation', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('docs/images/backpropagation.png', dpi=150, bbox_inches='tight')
    plt.show()

backpropagation_visualization()
```

## Training Process

### Loss Function and Optimization

```python
def training_visualization():
    """Visualize the training process"""
    # Simulated training loss
    epochs = np.arange(1, 101)
    train_loss = 2.0 * np.exp(-epochs/30) + 0.1 + np.random.normal(0, 0.05, 100)
    val_loss = 2.2 * np.exp(-epochs/35) + 0.15 + np.random.normal(0, 0.06, 100)
    
    train_acc = 1 - train_loss / 2.0 + np.random.normal(0, 0.02, 100)
    val_acc = 1 - val_loss / 2.0 + np.random.normal(0, 0.02, 100)
    train_acc = np.clip(train_acc, 0, 1)
    val_acc = np.clip(val_acc, 0, 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    axes[0].plot(epochs, val_loss, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, train_acc, 'b-', linewidth=2, label='Training Accuracy', alpha=0.8)
    axes[1].plot(epochs, val_acc, 'r-', linewidth=2, label='Validation Accuracy', alpha=0.8)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('docs/images/training_process.png', dpi=150, bbox_inches='tight')
    plt.show()

training_visualization()
```

## Activation Functions

Activation functions introduce non-linearity, enabling neural networks to learn complex patterns.

```python
def visualize_activation_functions():
    """Visualize common activation functions"""
    x = np.linspace(-5, 5, 1000)
    
    # Activation functions
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    relu = np.maximum(0, x)
    leaky_relu = np.where(x > 0, x, 0.01 * x)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Sigmoid
    axes[0, 0].plot(x, sigmoid, 'b-', linewidth=2)
    axes[0, 0].set_title('Sigmoid: σ(x) = 1/(1+e⁻ˣ)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('x', fontsize=11)
    axes[0, 0].set_ylabel('σ(x)', fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[0, 0].axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    axes[0, 0].set_ylim(-0.1, 1.1)
    
    # Tanh
    axes[0, 1].plot(x, tanh, 'g-', linewidth=2)
    axes[0, 1].set_title('Tanh: tanh(x)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('x', fontsize=11)
    axes[0, 1].set_ylabel('tanh(x)', fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[0, 1].axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    axes[0, 1].set_ylim(-1.1, 1.1)
    
    # ReLU
    axes[1, 0].plot(x, relu, 'r-', linewidth=2)
    axes[1, 0].set_title('ReLU: max(0, x)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('x', fontsize=11)
    axes[1, 0].set_ylabel('ReLU(x)', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[1, 0].axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    
    # Leaky ReLU
    axes[1, 1].plot(x, leaky_relu, 'orange', linewidth=2)
    axes[1, 1].set_title('Leaky ReLU: max(0.01x, x)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('x', fontsize=11)
    axes[1, 1].set_ylabel('Leaky ReLU(x)', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[1, 1].axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('docs/images/activation_functions.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_activation_functions()
```

### Activation Function Derivatives

```python
def visualize_activation_derivatives():
    """Visualize derivatives of activation functions"""
    x = np.linspace(-5, 5, 1000)
    
    # Derivatives
    sigmoid_deriv = (1 / (1 + np.exp(-x))) * (1 - 1 / (1 + np.exp(-x)))
    tanh_deriv = 1 - np.tanh(x)**2
    relu_deriv = np.where(x > 0, 1, 0)
    leaky_relu_deriv = np.where(x > 0, 1, 0.01)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(x, sigmoid_deriv, 'b-', linewidth=2)
    axes[0, 0].set_title("Sigmoid Derivative", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('x', fontsize=11)
    axes[0, 0].set_ylabel("σ'(x)", fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(x, tanh_deriv, 'g-', linewidth=2)
    axes[0, 1].set_title("Tanh Derivative", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('x', fontsize=11)
    axes[0, 1].set_ylabel("tanh'(x)", fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(x, relu_deriv, 'r-', linewidth=2)
    axes[1, 0].set_title("ReLU Derivative", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('x', fontsize=11)
    axes[1, 0].set_ylabel("ReLU'(x)", fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(x, leaky_relu_deriv, 'orange', linewidth=2)
    axes[1, 1].set_title("Leaky ReLU Derivative", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('x', fontsize=11)
    axes[1, 1].set_ylabel("Leaky ReLU'(x)", fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('docs/images/activation_derivatives.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_activation_derivatives()
```

## Practice Exercises

1. **Forward Propagation**: Given inputs `[1, 2]`, weights `[[0.5, 0.3], [0.2, 0.8]]`, bias `[0.1, 0.2]`, and ReLU activation, compute the output.

2. **Loss Calculation**: If predicted output is `0.7` and true value is `1.0`, calculate:
   - Mean Squared Error (MSE)
   - Binary Cross-Entropy Loss

3. **Gradient Descent**: Update weights using gradient `-0.5` and learning rate `0.01`.

## Summary

- **Perceptron** is the basic building block
- **MLPs** stack multiple layers for complex learning
- **Forward propagation** computes predictions
- **Backpropagation** computes gradients for learning
- **Activation functions** introduce non-linearity
- **Training** minimizes loss through gradient descent

## Next Steps

- Learn about [CNNs](05_cnns.md) for image processing
- Study [RNNs](06_rnns.md) for sequence data
