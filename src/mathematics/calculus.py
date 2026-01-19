"""
Calculus and Optimization - Comprehensive module
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import quad, trapz
from typing import Callable, Optional, Tuple


def numerical_derivative(f: Callable, x: float, h: float = 1e-5) -> float:
    """Compute numerical derivative using finite differences"""
    return (f(x + h) - f(x - h)) / (2 * h)


def visualize_function_and_derivative(f: Callable, df: Callable, 
                                     x_range: Tuple[float, float] = (-5, 5),
                                     save_path: Optional[str] = None):
    """Visualize function and its derivative"""
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = f(x)
    dy = df(x)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(x, y, 'b-', linewidth=2, label='f(x)')
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('f(x)', fontsize=12)
    axes[0].set_title('Function f(x)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    
    axes[1].plot(x, dy, 'r-', linewidth=2, label="f'(x)")
    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[1].axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel("f'(x)", fontsize=12)
    axes[1].set_title("Derivative f'(x)", fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def visualize_tangent_lines(f: Callable, df: Callable, points: list,
                           x_range: Tuple[float, float] = (-3, 3),
                           save_path: Optional[str] = None):
    """Visualize tangent lines at different points"""
    x_plot = np.linspace(x_range[0], x_range[1], 1000)
    y_plot = f(x_plot)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x)')
    
    for point in points:
        y_point = f(point)
        slope = df(point)
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
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def visualize_gradient_field(f_2d: Callable, x_range: Tuple[float, float] = (-3, 3),
                            y_range: Tuple[float, float] = (-3, 3),
                            save_path: Optional[str] = None):
    """Visualize gradient field of a 2D function"""
    x = np.linspace(x_range[0], x_range[1], 20)
    y = np.linspace(y_range[0], y_range[1], 20)
    X, Y = np.meshgrid(x, y)
    Z = f_2d(X, Y)
    
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
    
    print(f"Starting point: {x_history[0]}")
    print(f"Final point: {x_history[-1]}")
    print(f"Minimum value: {f_history[-1]}")
    print(f"True minimum at x = -1, f(-1) = {f(-1)}")
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
    """Compare different learning rates"""
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


def partial_derivatives_example():
    """Demonstrate partial derivatives"""
    def f(x, y):
        return x**2 + y**2 + 2*x*y
    def df_dx(x, y):
        return 2*x + 2*y
    def df_dy(x, y):
        return 2*y + 2*x
    def gradient(x, y):
        return np.array([df_dx(x, y), df_dy(x, y)])
    
    x, y = 1.0, 2.0
    print(f"Function: f(x, y) = x² + y² + 2xy")
    print(f"At point ({x}, {y}):")
    print(f"  f({x}, {y}) = {f(x, y)}")
    print(f"  ∂f/∂x = {df_dx(x, y)}")
    print(f"  ∂f/∂y = {df_dy(x, y)}")
    print(f"  Gradient = {gradient(x, y)}")
    return gradient


def integration_example():
    """Demonstrate numerical integration"""
    def f(x):
        return x**2
    analytical = 8/3
    numerical_quad, error = quad(f, 0, 2)
    x = np.linspace(0, 2, 100)
    y = f(x)
    numerical_trapz = trapz(y, x)
    
    print(f"Function: f(x) = x²")
    print(f"Integration from 0 to 2:")
    print(f"  Analytical: {analytical:.6f}")
    print(f"  Quad (scipy): {numerical_quad:.6f} (error: {error:.2e})")
    print(f"  Trapezoidal: {numerical_trapz:.6f}")
    return analytical, numerical_quad, numerical_trapz


def neural_network_gradient_example():
    """Demonstrate neural network gradient computation"""
    x, w, b, y_true = 2.0, 1.5, 0.5, 5.0
    y_pred = w * x + b
    loss = (y_pred - y_true)**2
    
    print("Forward Pass:")
    print(f"  Input: x = {x}")
    print(f"  Weights: w = {w}, b = {b}")
    print(f"  Prediction: y = {y_pred:.2f}")
    print(f"  True value: {y_true}")
    print(f"  Loss: L = {loss:.4f}")
    
    dL_dy = 2 * (y_pred - y_true)
    dy_dw = x
    dy_db = 1
    dL_dw = dL_dy * dy_dw
    dL_db = dL_dy * dy_db
    
    print("\nBackward Pass (Gradients):")
    print(f"  ∂L/∂y = {dL_dy:.4f}")
    print(f"  ∂L/∂w = {dL_dw:.4f}")
    print(f"  ∂L/∂b = {dL_db:.4f}")
    
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
    
    ax.arrow(1.3, 1, 1.2, 0.8, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(1.3, 3, 1.2, -0.8, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(3.3, 2, 1.2, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.text(2.2, 1.5, '×', fontsize=16, fontweight='bold')
    ax.text(2.2, 2.5, '×', fontsize=16, fontweight='bold')
    ax.text(4.2, 2, 'loss', fontsize=12)
    
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
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


if __name__ == "__main__":
    print("=== Gradient Descent ===")
    gradient_descent_example()
    print("\n=== Partial Derivatives ===")
    partial_derivatives_example()
    print("\n=== Numerical Integration ===")
    integration_example()
    print("\n=== Neural Network Gradients ===")
    neural_network_gradient_example()
