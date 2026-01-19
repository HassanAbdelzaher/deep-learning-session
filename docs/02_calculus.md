# Calculus for Deep Learning

## Table of Contents
1. [Introduction](#introduction)
2. [Derivatives](#derivatives)
3. [Gradients](#gradients)
4. [Optimization](#optimization)
5. [Backpropagation Connection](#backpropagation-connection)

## Introduction

Calculus is essential for understanding how neural networks learn. The training process uses:
- **Derivatives** to measure how functions change
- **Gradients** to find the direction of steepest ascent/descent
- **Optimization** to minimize loss functions

## Derivatives

### What is a Derivative?

The derivative of a function `f(x)` at point `x` measures the **instantaneous rate of change** or the slope of the tangent line.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative

# Define a function
def f(x):
    return x**2 + 2*x + 1

# Analytical derivative
def df(x):
    return 2*x + 2

# Visualize function and its derivative
x = np.linspace(-5, 5, 1000)
y = f(x)
dy = df(x)

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Function
axes[0].plot(x, y, 'b-', linewidth=2, label='f(x) = x² + 2x + 1')
axes[0].set_xlabel('x', fontsize=12)
axes[0].set_ylabel('f(x)', fontsize=12)
axes[0].set_title('Function f(x)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10)

# Derivative
axes[1].plot(x, dy, 'r-', linewidth=2, label="f'(x) = 2x + 2")
axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
axes[1].axvline(x=0, color='k', linestyle='--', linewidth=0.5)
axes[1].set_xlabel('x', fontsize=12)
axes[1].set_ylabel("f'(x)", fontsize=12)
axes[1].set_title("Derivative f'(x)", fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('docs/images/function_and_derivative.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Tangent Lines

The derivative gives us the slope of the tangent line at any point.

```python
# Visualize tangent lines at different points
x_plot = np.linspace(-3, 3, 1000)
y_plot = f(x_plot)

# Points to show tangents
points = [-2, 0, 2]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x) = x² + 2x + 1')

for point in points:
    # Function value at point
    y_point = f(point)
    
    # Slope (derivative) at point
    slope = df(point)
    
    # Tangent line: y = slope * (x - point) + y_point
    x_tangent = np.linspace(point - 1.5, point + 1.5, 100)
    y_tangent = slope * (x_tangent - point) + y_point
    
    ax.plot(x_tangent, y_tangent, 'r--', linewidth=1.5, alpha=0.7)
    ax.plot(point, y_point, 'ro', markersize=8)
    ax.text(point + 0.2, y_point + 0.5, f'x={point}\nslope={slope:.1f}', 
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('f(x)', fontsize=12)
ax.set_title('Function with Tangent Lines', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('docs/images/tangent_lines.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Key Insight**: Where the derivative is zero, the function has a minimum or maximum (critical point).

## Gradients

### Gradient of a Function

For a function `f(x, y)`, the gradient `∇f` is a vector of partial derivatives:
```
∇f = [∂f/∂x, ∂f/∂y]
```

The gradient points in the direction of **steepest ascent**.

```python
# Function: f(x, y) = x² + y²
def f_2d(x, y):
    return x**2 + y**2

def gradient_f(x, y):
    return np.array([2*x, 2*y])

# Create a grid
x = np.linspace(-3, 3, 20)
y = np.linspace(-3, 3, 20)
X, Y = np.meshgrid(x, y)
Z = f_2d(X, Y)

# Calculate gradients at grid points
U, V = np.gradient(Z)
U = U / 10  # Normalize for visualization
V = V / 10

# Visualize
fig, ax = plt.subplots(figsize=(10, 8))

# Contour plot
contour = ax.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.6)
ax.clabel(contour, inline=True, fontsize=8)

# Gradient field (vector field)
ax.quiver(X, Y, U, V, scale=30, width=0.003, color='red', alpha=0.7)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Gradient Field of f(x,y) = x² + y²', fontsize=14, fontweight='bold')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.colorbar(contour, ax=ax, label='f(x,y)')
plt.tight_layout()
plt.savefig('docs/images/gradient_field.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Key Insight**: The gradient always points toward higher values of the function. To minimize, we move in the **opposite direction** (negative gradient).

## Optimization

### Gradient Descent

Gradient descent is the algorithm used to train neural networks. It finds the minimum of a function by:
1. Starting at a random point
2. Computing the gradient
3. Moving in the negative gradient direction
4. Repeating until convergence

```python
def gradient_descent_visualization():
    # Function: f(x) = x² + 2x + 1 (minimum at x = -1)
    def f(x):
        return x**2 + 2*x + 1
    
    def df(x):
        return 2*x + 2
    
    # Gradient descent parameters
    learning_rate = 0.1
    x = 3.0  # Starting point
    iterations = 30
    
    # Store history
    x_history = [x]
    f_history = [f(x)]
    
    # Perform gradient descent
    for i in range(iterations):
        gradient = df(x)
        x = x - learning_rate * gradient
        x_history.append(x)
        f_history.append(f(x))
    
    # Visualize
    x_plot = np.linspace(-4, 4, 1000)
    y_plot = f(x_plot)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Function with path
    axes[0].plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x) = x² + 2x + 1')
    axes[0].plot(x_history, f_history, 'ro-', linewidth=1.5, markersize=6, 
                 label='Gradient Descent Path', alpha=0.7)
    axes[0].axvline(x=-1, color='g', linestyle='--', linewidth=1.5, 
                    label='True Minimum (x=-1)', alpha=0.7)
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('f(x)', fontsize=12)
    axes[0].set_title('Gradient Descent Optimization', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    
    # Loss over iterations
    axes[1].plot(range(len(f_history)), f_history, 'b-o', linewidth=2, markersize=6)
    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('f(x)', fontsize=12)
    axes[1].set_title('Loss Convergence', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('docs/images/gradient_descent.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Starting point: {x_history[0]:.2f}")
    print(f"Final point: {x_history[-1]:.2f}")
    print(f"True minimum: -1.00")
    print(f"Final loss: {f_history[-1]:.6f}")

gradient_descent_visualization()
```

### Learning Rate Effect

The learning rate controls how big steps we take. Too large → overshoot; too small → slow convergence.

```python
def compare_learning_rates():
    def f(x):
        return x**2 + 2*x + 1
    
    def df(x):
        return 2*x + 2
    
    learning_rates = [0.01, 0.1, 0.5, 1.0]
    colors = ['blue', 'green', 'orange', 'red']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, lr in enumerate(learning_rates):
        x = 3.0
        x_history = [x]
        f_history = [f(x)]
        
        for i in range(30):
            gradient = df(x)
            x = x - lr * gradient
            x_history.append(x)
            f_history.append(f(x))
        
        x_plot = np.linspace(-4, 4, 1000)
        y_plot = f(x_plot)
        
        axes[idx].plot(x_plot, y_plot, 'b-', linewidth=1.5, alpha=0.5)
        axes[idx].plot(x_history, f_history, 'o-', linewidth=1.5, 
                      markersize=4, color=colors[idx], label=f'LR={lr}')
        axes[idx].axvline(x=-1, color='g', linestyle='--', linewidth=1, alpha=0.5)
        axes[idx].set_xlabel('x', fontsize=10)
        axes[idx].set_ylabel('f(x)', fontsize=10)
        axes[idx].set_title(f'Learning Rate = {lr}', fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig('docs/images/learning_rates.png', dpi=150, bbox_inches='tight')
    plt.show()

compare_learning_rates()
```

## Backpropagation Connection

### How Calculus Enables Backpropagation

In neural networks:
1. **Forward pass**: Compute predictions and loss
2. **Backward pass**: Compute gradients using chain rule
3. **Update**: Move weights in negative gradient direction

```python
# Simplified neural network gradient example
def neural_network_gradient_example():
    # Simple network: y = w * x + b
    # Loss: L = (y_pred - y_true)²
    
    # Given
    x = 2.0
    w = 1.5
    b = 0.5
    y_true = 5.0
    
    # Forward pass
    y_pred = w * x + b
    loss = (y_pred - y_true)**2
    
    print("Forward Pass:")
    print(f"  Input: x = {x}")
    print(f"  Weights: w = {w}, b = {b}")
    print(f"  Prediction: y = {y_pred:.2f}")
    print(f"  True value: {y_true}")
    print(f"  Loss: L = {loss:.4f}")
    
    # Backward pass (chain rule)
    dL_dy = 2 * (y_pred - y_true)  # ∂L/∂y
    dy_dw = x                       # ∂y/∂w
    dy_db = 1                       # ∂y/∂b
    
    dL_dw = dL_dy * dy_dw           # ∂L/∂w = ∂L/∂y × ∂y/∂w
    dL_db = dL_dy * dy_db           # ∂L/∂b = ∂L/∂y × ∂y/∂b
    
    print("\nBackward Pass (Gradients):")
    print(f"  ∂L/∂y = {dL_dy:.4f}")
    print(f"  ∂L/∂w = {dL_dw:.4f}")
    print(f"  ∂L/∂b = {dL_db:.4f}")
    
    # Update weights
    learning_rate = 0.1
    w_new = w - learning_rate * dL_dw
    b_new = b - learning_rate * dL_db
    
    print("\nWeight Update:")
    print(f"  w: {w:.4f} → {w_new:.4f}")
    print(f"  b: {b:.4f} → {b_new:.4f}")
    
    # Verify improvement
    y_pred_new = w_new * x + b_new
    loss_new = (y_pred_new - y_true)**2
    print(f"\nNew prediction: {y_pred_new:.4f}")
    print(f"New loss: {loss_new:.4f} (was {loss:.4f})")

neural_network_gradient_example()
```

### Chain Rule Visualization

```python
# Visualize how gradients flow backward
fig, ax = plt.subplots(figsize=(12, 8))

# Simple computation graph: x → (×w) → y → (loss)
nodes = {
    'x': (1, 3),
    'w': (1, 1),
    'y': (3, 2),
    'L': (5, 2)
}

# Draw nodes
for name, (x, y) in nodes.items():
    circle = plt.Circle((x, y), 0.3, color='lightblue', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(x, y, name, ha='center', va='center', fontsize=14, fontweight='bold')

# Draw arrows
ax.arrow(1.3, 1, 1.2, 0.8, head_width=0.1, head_length=0.1, fc='black', ec='black')
ax.arrow(1.3, 3, 1.2, -0.8, head_width=0.1, head_length=0.1, fc='black', ec='black')
ax.arrow(3.3, 2, 1.2, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

# Labels
ax.text(2.2, 1.5, '×', fontsize=16, fontweight='bold')
ax.text(2.2, 2.5, '×', fontsize=16, fontweight='bold')
ax.text(4.2, 2, 'loss', fontsize=12)

# Gradient flow (backward)
ax.arrow(4.7, 2, -1.2, 0, head_width=0.1, head_length=0.1, 
         fc='red', ec='red', linewidth=2, linestyle='--')
ax.arrow(3.3, 2, -1.2, 0.8, head_width=0.1, head_length=0.1, 
         fc='red', ec='red', linewidth=2, linestyle='--')
ax.arrow(3.3, 2, -1.2, -0.8, head_width=0.1, head_length=0.1, 
         fc='red', ec='red', linewidth=2, linestyle='--')

ax.text(4.5, 2.3, '∂L/∂y', fontsize=10, color='red', fontweight='bold')
ax.text(2.5, 2.8, '∂L/∂w', fontsize=10, color='red', fontweight='bold')
ax.text(2.5, 1.2, '∂L/∂x', fontsize=10, color='red', fontweight='bold')

ax.set_xlim(0, 6)
ax.set_ylim(0, 4)
ax.set_title('Computation Graph and Gradient Flow', fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig('docs/images/computation_graph.png', dpi=150, bbox_inches='tight')
plt.show()
```

## Practice Exercises

1. **Derivative**: Find the derivative of `f(x) = 3x² + 5x - 2` at `x = 2`.

2. **Gradient**: For `f(x, y) = x²y + xy²`, compute the gradient at `(1, 2)`.

3. **Gradient Descent**: Implement gradient descent to minimize `f(x) = (x - 3)²` starting from `x = 0`.

## Summary

- **Derivatives** measure rate of change
- **Gradients** point toward steepest ascent
- **Gradient descent** minimizes functions by moving opposite to gradient
- **Chain rule** enables backpropagation in neural networks
- **Learning rate** controls step size in optimization

## Next Steps

- Study [Statistics for Deep Learning](03_statistics.md)
- Learn about [Neural Networks](04_neural_networks.md)
