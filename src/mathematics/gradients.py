"""
Gradients and Gradient-Based Optimization
Comprehensive module for understanding gradients, gradient descent, and optimization
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Callable, Optional, Tuple, List

# Fix Unicode encoding for Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')


# ============================================================================
# GRADIENT COMPUTATION
# ============================================================================

def compute_gradient_1d(f: Callable, x: float, h: float = 1e-5) -> float:
    """Compute gradient (derivative) of 1D function using finite differences"""
    return (f(x + h) - f(x - h)) / (2 * h)


def compute_gradient_2d(f: Callable, x: float, y: float, h: float = 1e-5) -> np.ndarray:
    """Compute gradient of 2D function: ∇f = [∂f/∂x, ∂f/∂y]"""
    df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return np.array([df_dx, df_dy])


def partial_derivatives_example():
    """Demonstrate partial derivatives and gradient computation"""
    def f(x, y):
        return x**2 + y**2 + 2*x*y
    
    def df_dx(x, y):
        return 2*x + 2*y
    
    def df_dy(x, y):
        return 2*y + 2*x
    
    def gradient(x, y):
        return np.array([df_dx(x, y), df_dy(x, y)])
    
    x, y = 1.0, 2.0
    print(f"Function: f(x, y) = x^2 + y^2 + 2xy")
    print(f"At point ({x}, {y}):")
    print(f"  f({x}, {y}) = {f(x, y)}")
    print(f"  df/dx = {df_dx(x, y)}")
    print(f"  df/dy = {df_dy(x, y)}")
    print(f"  Gradient = {gradient(x, y)}")
    print(f"  Gradient magnitude = {np.linalg.norm(gradient(x, y)):.4f}")
    print(f"  Gradient direction = {gradient(x, y) / np.linalg.norm(gradient(x, y))}")
    return gradient


# ============================================================================
# GRADIENT VISUALIZATION
# ============================================================================

def visualize_gradient_field(f_2d: Callable, x_range: Tuple[float, float] = (-3, 3),
                            y_range: Tuple[float, float] = (-3, 3),
                            save_path: Optional[str] = None):
    """Visualize gradient field of a 2D function"""
    x = np.linspace(x_range[0], x_range[1], 20)
    y = np.linspace(y_range[0], y_range[1], 20)
    X, Y = np.meshgrid(x, y)
    Z = f_2d(X, Y)
    
    # Compute gradients
    U, V = np.gradient(Z)
    U, V = U / 10, V / 10
    
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8)
    ax.quiver(X, Y, U, V, scale=30, width=0.003, color='red', alpha=0.7)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Gradient Field', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.colorbar(contour, ax=ax, label='f(x,y)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def visualize_gradient_direction(f: Callable, df: Callable, point: float,
                                x_range: Tuple[float, float] = (-3, 3),
                                save_path: Optional[str] = None):
    """Visualize gradient direction at a specific point"""
    x_plot = np.linspace(x_range[0], x_range[1], 1000)
    y_plot = f(x_plot)
    
    # Point and gradient
    y_point = f(point)
    gradient = df(point)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x)')
    ax.plot(point, y_point, 'ro', markersize=10, label=f'Point: x={point}')
    
    # Gradient direction (arrow)
    if gradient > 0:
        ax.arrow(point, y_point, 0.5, gradient * 0.5, head_width=0.1, 
                head_length=0.1, fc='red', ec='red', linewidth=2, 
                label=f'Gradient: {gradient:.2f} (upward)')
    else:
        ax.arrow(point, y_point, -0.5, gradient * 0.5, head_width=0.1, 
                head_length=0.1, fc='red', ec='red', linewidth=2, 
                label=f'Gradient: {gradient:.2f} (downward)')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title('Gradient Direction at a Point', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ============================================================================
# GRADIENT DESCENT
# ============================================================================

def gradient_descent_example():
    """Demonstrate gradient descent optimization"""
    def f(x):
        return x**2 + 2*x + 1
    def df(x):
        return 2*x + 2
    
    learning_rate = 0.1
    x = 5.0
    iterations = 50
    x_history = [x]
    f_history = [f(x)]
    
    for i in range(iterations):
        gradient = df(x)
        x = x - learning_rate * gradient
        x_history.append(x)
        f_history.append(f(x))
    
    print(f"Starting point: {x_history[0]:.4f}")
    print(f"Final point: {x_history[-1]:.4f}")
    print(f"Minimum value: {f_history[-1]:.6f}")
    print(f"True minimum at x = -1, f(-1) = {f(-1)}")
    print(f"Converged: {abs(x_history[-1] + 1) < 0.01}")
    return x_history, f_history


def visualize_gradient_descent(f: Callable, df: Callable, x_start: float,
                              learning_rate: float, iterations: int,
                              x_range: Tuple[float, float] = (-4, 4),
                              save_path: Optional[str] = None):
    """Visualize gradient descent optimization"""
    x = x_start
    x_history = [x]
    f_history = [f(x)]
    
    for i in range(iterations):
        gradient = df(x)
        x = x - learning_rate * gradient
        x_history.append(x)
        f_history.append(f(x))
    
    x_plot = np.linspace(x_range[0], x_range[1], 1000)
    y_plot = f(x_plot)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    axes[0].plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x)')
    axes[0].plot(x_history, f_history, 'ro-', linewidth=1.5, markersize=6, 
                 label='Gradient Descent Path', alpha=0.7)
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('f(x)', fontsize=12)
    axes[0].set_title('Gradient Descent Optimization', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    
    axes[1].plot(range(len(f_history)), f_history, 'b-o', linewidth=2, markersize=6)
    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('f(x)', fontsize=12)
    axes[1].set_title('Loss Convergence', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, x_history, f_history


def compare_learning_rates(f: Callable, df: Callable, x_start: float,
                          learning_rates: list, iterations: int = 30,
                          x_range: Tuple[float, float] = (-4, 4),
                          save_path: Optional[str] = None):
    """Compare different learning rates in gradient descent"""
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, lr in enumerate(learning_rates):
        x = x_start
        x_history = [x]
        f_history = [f(x)]
        
        for i in range(iterations):
            gradient = df(x)
            x = x - lr * gradient
            x_history.append(x)
            f_history.append(f(x))
        
        x_plot = np.linspace(x_range[0], x_range[1], 1000)
        y_plot = f(x_plot)
        axes[idx].plot(x_plot, y_plot, 'b-', linewidth=1.5, alpha=0.5)
        axes[idx].plot(x_history, f_history, 'o-', linewidth=1.5, 
                      markersize=4, color=colors[idx % len(colors)], label=f'LR={lr}')
        axes[idx].set_xlabel('x', fontsize=10)
        axes[idx].set_ylabel('f(x)', fontsize=10)
        axes[idx].set_title(f'Learning Rate = {lr}', fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend(fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def gradient_descent_2d(f_2d: Callable, df_dx: Callable, df_dy: Callable,
                       x_start: float, y_start: float, learning_rate: float,
                       iterations: int, save_path: Optional[str] = None):
    """Perform gradient descent on a 2D function"""
    x, y = x_start, y_start
    x_history = [x]
    y_history = [y]
    f_history = [f_2d(x, y)]
    
    for i in range(iterations):
        grad_x = df_dx(x, y)
        grad_y = df_dy(x, y)
        x = x - learning_rate * grad_x
        y = y - learning_rate * grad_y
        x_history.append(x)
        y_history.append(y)
        f_history.append(f_2d(x, y))
    
    # Visualize
    x_range = (min(x_history) - 1, max(x_history) + 1)
    y_range = (min(y_history) - 1, max(y_history) + 1)
    x_plot = np.linspace(x_range[0], x_range[1], 50)
    y_plot = np.linspace(y_range[0], y_range[1], 50)
    X, Y = np.meshgrid(x_plot, y_plot)
    Z = f_2d(X, Y)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8)
    ax.plot(x_history, y_history, 'ro-', linewidth=2, markersize=6, 
            label='Gradient Descent Path', alpha=0.7)
    ax.plot(x_history[0], y_history[0], 'go', markersize=10, label='Start')
    ax.plot(x_history[-1], y_history[-1], 'bo', markersize=10, label='End')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('2D Gradient Descent', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.colorbar(contour, ax=ax, label='f(x,y)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, x_history, y_history, f_history


# ============================================================================
# NEURAL NETWORK GRADIENTS
# ============================================================================

def neural_network_gradient_example():
    """Demonstrate neural network gradient computation (backpropagation)"""
    x, w, b, y_true = 2.0, 1.5, 0.5, 5.0
    y_pred = w * x + b
    loss = (y_pred - y_true)**2
    
    print("Forward Pass:")
    print(f"  Input: x = {x}")
    print(f"  Weights: w = {w}, b = {b}")
    print(f"  Prediction: y = {y_pred:.2f}")
    print(f"  True value: {y_true}")
    print(f"  Loss: L = {loss:.4f}")
    
    # Backward pass (gradients)
    dL_dy = 2 * (y_pred - y_true)
    dy_dw = x
    dy_db = 1
    dL_dw = dL_dy * dy_dw
    dL_db = dL_dy * dy_db
    
    print("\nBackward Pass (Gradients):")
    print(f"  dL/dy = {dL_dy:.4f}")
    print(f"  dL/dw = {dL_dw:.4f}")
    print(f"  dL/db = {dL_db:.4f}")
    
    learning_rate = 0.1
    w_new = w - learning_rate * dL_dw
    b_new = b - learning_rate * dL_db
    
    print("\nWeight Update:")
    print(f"  w: {w:.4f} → {w_new:.4f}")
    print(f"  b: {b:.4f} → {b_new:.4f}")
    
    y_pred_new = w_new * x + b_new
    loss_new = (y_pred_new - y_true)**2
    print(f"\nNew prediction: {y_pred_new:.4f}")
    print(f"New loss: {loss_new:.4f} (was {loss:.4f})")
    return dL_dw, dL_db


def visualize_computation_graph(save_path: Optional[str] = None):
    """Visualize computation graph and gradient flow"""
    from matplotlib.patches import Circle, Rectangle
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    nodes = {'x': (1, 3), 'w': (1, 1), 'y': (3, 2), 'L': (5, 2)}
    
    for name, (x, y) in nodes.items():
        if name == 'L':
            rect = Rectangle((x-0.3, y-0.2), 0.6, 0.4, color='lightcoral', 
                            ec='black', linewidth=2)
            ax.add_patch(rect)
        else:
            circle = Circle((x, y), 0.3, color='lightblue', ec='black', linewidth=2)
            ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Forward pass (black arrows)
    ax.arrow(1.3, 1, 1.2, 0.8, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(1.3, 3, 1.2, -0.8, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(3.3, 2, 1.2, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.text(2.2, 1.5, '×', fontsize=16, fontweight='bold')
    ax.text(2.2, 2.5, '×', fontsize=16, fontweight='bold')
    ax.text(4.2, 2, 'loss', fontsize=12)
    
    # Backward pass (red arrows - gradients)
    ax.arrow(4.7, 2, -1.2, 0, head_width=0.1, head_length=0.1, 
             fc='red', ec='red', linewidth=2, linestyle='--')
    ax.arrow(3.3, 2, -1.2, 0.8, head_width=0.1, head_length=0.1, 
             fc='red', ec='red', linewidth=2, linestyle='--')
    ax.arrow(3.3, 2, -1.2, -0.8, head_width=0.1, head_length=0.1, 
             fc='red', ec='red', linewidth=2, linestyle='--')
    
    ax.text(4.5, 2.3, 'dL/dy', fontsize=10, color='red', fontweight='bold')
    ax.text(2.5, 2.8, 'dL/dw', fontsize=10, color='red', fontweight='bold')
    ax.text(2.5, 1.2, 'dL/dx', fontsize=10, color='red', fontweight='bold')
    
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 4)
    ax.set_title('Computation Graph and Gradient Flow', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=== Gradients and Gradient-Based Optimization ===\n")
    
    print("=== Partial Derivatives and Gradient ===")
    partial_derivatives_example()
    
    print("\n=== Gradient Descent ===")
    gradient_descent_example()
    
    print("\n=== Neural Network Gradients ===")
    neural_network_gradient_example()
    
    print("\n=== Generating Visualizations ===")
    print("Run individual visualization functions to see graphs!")
    print("Example: visualize_gradient_descent(...)")
