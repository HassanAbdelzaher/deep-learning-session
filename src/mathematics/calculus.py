"""
Calculus and Optimization - Comprehensive module
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import quad
# trapz is in numpy, not scipy
from numpy import trapz
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


# Gradient functions have been moved to src/mathematics/gradients.py
# Import them from there if needed:
# from src.mathematics.gradients import (
#     visualize_gradient_field,
#     gradient_descent_example,
#     visualize_gradient_descent,
#     compare_learning_rates,
#     partial_derivatives_example,
#     neural_network_gradient_example,
#     visualize_computation_graph
# )


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


# Neural network gradient and computation graph functions moved to gradients.py


if __name__ == "__main__":
    print("=== Numerical Integration ===")
    integration_example()
    print("\nNote: Gradient-related examples have been moved to src/mathematics/gradients.py")
    print("Run 'python src/mathematics/gradients.py' to see gradient examples")
