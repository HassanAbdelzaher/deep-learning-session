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

### Single Perceptron

A perceptron is the simplest neural network - a single neuron.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.patches as mpatches

def visualize_perceptron():
    """Visualize a single perceptron"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Input nodes
    inputs = ['x₁', 'x₂', 'x₃']
    input_positions = [(1, 3), (1, 2), (1, 1)]
    
    for i, (label, pos) in enumerate(zip(inputs, input_positions)):
        circle = Circle(pos, 0.3, color='lightblue', ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], label, ha='center', va='center', 
               fontsize=14, fontweight='bold')
    
    # Neuron
    neuron_pos = (3, 2)
    neuron = Circle(neuron_pos, 0.4, color='lightgreen', ec='black', linewidth=2)
    ax.add_patch(neuron)
    ax.text(neuron_pos[0], neuron_pos[1], 'Σ', ha='center', va='center',
           fontsize=16, fontweight='bold')
    
    # Output
    output_pos = (5, 2)
    output = Circle(output_pos, 0.3, color='lightcoral', ec='black', linewidth=2)
    ax.add_patch(output)
    ax.text(output_pos[0], output_pos[1], 'y', ha='center', va='center',
           fontsize=14, fontweight='bold')
    
    # Weights
    weights = ['w₁', 'w₂', 'w₃']
    for i, (pos, weight) in enumerate(zip(input_positions, weights)):
        arrow = FancyArrowPatch((pos[0]+0.3, pos[1]), (neuron_pos[0]-0.4, neuron_pos[1]),
                               arrowstyle='->', mutation_scale=20, linewidth=1.5, color='blue')
        ax.add_patch(arrow)
        mid_x = (pos[0] + neuron_pos[0]) / 2
        mid_y = pos[1] + 0.2
        ax.text(mid_x, mid_y, weight, fontsize=10, ha='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Output arrow
    arrow = FancyArrowPatch((neuron_pos[0]+0.4, neuron_pos[1]), (output_pos[0]-0.3, output_pos[1]),
                           arrowstyle='->', mutation_scale=20, linewidth=2, color='red')
    ax.add_patch(arrow)
    
    # Formula
    ax.text(3, 0.3, 'y = f(w₁x₁ + w₂x₂ + w₃x₃ + b)', ha='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 4)
    ax.set_title('Single Perceptron', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('docs/images/perceptron.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_perceptron()
```

### Perceptron Decision Boundary

```python
def perceptron_decision_boundary():
    """Show how a perceptron creates a decision boundary"""
    # Generate data
    np.random.seed(42)
    class0 = np.random.randn(50, 2) + [1, 1]
    class1 = np.random.randn(50, 2) + [3, 3]
    
    # Simple perceptron: y = sign(w1*x1 + w2*x2 + b)
    w1, w2, b = -1, 1, 0.5
    
    # Decision boundary: w1*x1 + w2*x2 + b = 0
    # Solving for x2: x2 = -(w1*x1 + b) / w2
    x1_boundary = np.linspace(-1, 5, 100)
    x2_boundary = -(w1 * x1_boundary + b) / w2
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot data
    ax.scatter(class0[:, 0], class0[:, 1], c='blue', s=100, alpha=0.6, 
              label='Class 0', edgecolors='black', linewidth=1)
    ax.scatter(class1[:, 0], class1[:, 1], c='red', s=100, alpha=0.6,
              label='Class 1', edgecolors='black', linewidth=1)
    
    # Decision boundary
    ax.plot(x1_boundary, x2_boundary, 'g-', linewidth=3, label='Decision Boundary')
    
    # Fill regions
    ax.fill_between(x1_boundary, x2_boundary, 6, alpha=0.2, color='blue', label='Class 0 Region')
    ax.fill_between(x1_boundary, x2_boundary, -2, alpha=0.2, color='red', label='Class 1 Region')
    
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('Perceptron Decision Boundary', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 5)
    plt.tight_layout()
    plt.savefig('docs/images/perceptron_boundary.png', dpi=150, bbox_inches='tight')
    plt.show()

perceptron_decision_boundary()
```

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
