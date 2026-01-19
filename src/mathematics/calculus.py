"""
Calculus and Optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def numerical_derivative(f, x, h=1e-5):
    """
    Compute numerical derivative using finite differences
    
    Args:
        f: function to differentiate
        x: point at which to compute derivative
        h: step size
    
    Returns:
        derivative value
    """
    return (f(x + h) - f(x - h)) / (2 * h)


def gradient_descent_example():
    """Demonstrate gradient descent optimization"""
    # Define a simple quadratic function: f(x) = x^2 + 2x + 1
    def f(x):
        return x**2 + 2*x + 1
    
    def df(x):
        return 2*x + 2
    
    # Gradient descent parameters
    learning_rate = 0.1
    x = 5.0  # Starting point
    iterations = 50
    
    # Store history
    x_history = [x]
    f_history = [f(x)]
    
    # Perform gradient descent
    for i in range(iterations):
        gradient = df(x)
        x = x - learning_rate * gradient
        x_history.append(x)
        f_history.append(f(x))
    
    print(f"Starting point: {x_history[0]}")
    print(f"Final point: {x_history[-1]}")
    print(f"Minimum value: {f_history[-1]}")
    print(f"True minimum at x = -1, f(-1) = {f(-1)}")
    
    # Visualize
    x_plot = np.linspace(-3, 6, 100)
    y_plot = f(x_plot)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot, 'b-', label='f(x) = x² + 2x + 1', linewidth=2)
    plt.plot(x_history, f_history, 'ro-', label='Gradient Descent Path', markersize=4)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Gradient Descent Optimization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return x_history, f_history


def partial_derivatives_example():
    """Demonstrate partial derivatives"""
    # Function: f(x, y) = x^2 + y^2 + 2xy
    def f(x, y):
        return x**2 + y**2 + 2*x*y
    
    def df_dx(x, y):
        return 2*x + 2*y
    
    def df_dy(x, y):
        return 2*y + 2*x
    
    def gradient(x, y):
        return np.array([df_dx(x, y), df_dy(x, y)])
    
    # Test at a point
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
    from scipy.integrate import quad, trapz
    
    # Function to integrate: f(x) = x^2
    def f(x):
        return x**2
    
    # Analytical integral from 0 to 2: ∫x²dx = x³/3 = 8/3 ≈ 2.667
    analytical = 8/3
    
    # Numerical integration using quad
    numerical_quad, error = quad(f, 0, 2)
    
    # Numerical integration using trapezoidal rule
    x = np.linspace(0, 2, 100)
    y = f(x)
    numerical_trapz = trapz(y, x)
    
    print(f"Function: f(x) = x²")
    print(f"Integration from 0 to 2:")
    print(f"  Analytical: {analytical:.6f}")
    print(f"  Quad (scipy): {numerical_quad:.6f} (error: {error:.2e})")
    print(f"  Trapezoidal: {numerical_trapz:.6f}")
    
    return analytical, numerical_quad, numerical_trapz


if __name__ == "__main__":
    print("=== Gradient Descent ===")
    gradient_descent_example()
    
    print("\n=== Partial Derivatives ===")
    partial_derivatives_example()
    
    print("\n=== Numerical Integration ===")
    integration_example()
