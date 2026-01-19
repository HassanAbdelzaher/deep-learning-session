"""
Neural Networks Fundamentals
Comprehensive module with visualizations and examples
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Optional
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import matplotlib.patches as mpatches


class SimpleNeuralNetwork:
    """A simple feedforward neural network from scratch"""
    
    def __init__(self, layers: List[int], learning_rate: float = 0.01):
        """
        Initialize neural network
        
        Args:
            layers: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
            learning_rate: Learning rate for gradient descent
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layers) - 1):
            # Xavier initialization
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid"""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Forward propagation
        
        Returns:
            activations: List of activations for each layer
            z_values: List of pre-activation values
        """
        activations = [X]
        z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            # Use sigmoid for hidden layers, sigmoid for output
            if i < len(self.weights) - 1:
                a = self.relu(z)  # ReLU for hidden layers
            else:
                a = self.sigmoid(z)  # Sigmoid for output layer
            activations.append(a)
        
        return activations, z_values
    
    def backward(self, X: np.ndarray, y: np.ndarray, activations: List[np.ndarray], 
                 z_values: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backward propagation
        
        Returns:
            weight_gradients: Gradients for weights
            bias_gradients: Gradients for biases
        """
        m = X.shape[0]
        weight_gradients = [np.zeros_like(w) for w in self.weights]
        bias_gradients = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error
        delta = activations[-1] - y
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Calculate gradients
            weight_gradients[i] = (1/m) * np.dot(activations[i].T, delta)
            bias_gradients[i] = (1/m) * np.sum(delta, axis=0, keepdims=True)
            
            # Propagate error to previous layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                # Apply derivative of activation function
                delta *= self.relu_derivative(z_values[i-1])
        
        return weight_gradients, bias_gradients
    
    def update_weights(self, weight_gradients: List[np.ndarray], 
                      bias_gradients: List[np.ndarray]):
        """Update weights and biases using gradients"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
              verbose: bool = True) -> List[float]:
        """
        Train the neural network
        
        Returns:
            loss_history: List of loss values during training
        """
        loss_history = []
        
        for epoch in range(epochs):
            # Forward pass
            activations, z_values = self.forward(X)
            
            # Calculate loss (MSE)
            loss = np.mean((activations[-1] - y) ** 2)
            loss_history.append(loss)
            
            # Backward pass
            weight_gradients, bias_gradients = self.backward(X, y, activations, z_values)
            
            # Update weights
            self.update_weights(weight_gradients, bias_gradients)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
        
        return loss_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        activations, _ = self.forward(X)
        return activations[-1]


def xor_problem_example():
    """Demonstrate neural network solving XOR problem"""
    # XOR problem data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Create and train network
    nn = SimpleNeuralNetwork(layers=[2, 4, 1], learning_rate=0.5)
    print("Training neural network on XOR problem...")
    loss_history = nn.train(X, y, epochs=2000, verbose=True)
    
    # Make predictions
    predictions = nn.predict(X)
    print("\nPredictions:")
    for i, (input_val, target, pred) in enumerate(zip(X, y, predictions)):
        print(f"Input: {input_val}, Target: {target[0]}, Prediction: {pred[0]:.4f}")
    
    # Plot loss history
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (XOR Problem)')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    
    return nn, loss_history


def linear_regression_example():
    """Demonstrate neural network for linear regression"""
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = 2 * X + 1 + np.random.randn(100, 1) * 0.5
    
    # Create and train network
    nn = SimpleNeuralNetwork(layers=[1, 5, 1], learning_rate=0.01)
    print("Training neural network for linear regression...")
    loss_history = nn.train(X, y, epochs=500, verbose=True)
    
    # Make predictions
    X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    y_pred = nn.predict(X_test)
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.6, label='Training Data')
    plt.plot(X_test, y_pred, 'r-', linewidth=2, label='Neural Network Prediction')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Neural Network Linear Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return nn, loss_history


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_perceptron(save_path: Optional[str] = None):
    """Visualize a single perceptron"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    inputs = ['x₁', 'x₂', 'x₃']
    input_positions = [(1, 3), (1, 2), (1, 1)]
    
    for i, (label, pos) in enumerate(zip(inputs, input_positions)):
        circle = Circle(pos, 0.3, color='lightblue', ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], label, ha='center', va='center', 
               fontsize=14, fontweight='bold')
    
    neuron_pos = (3, 2)
    neuron = Circle(neuron_pos, 0.4, color='lightgreen', ec='black', linewidth=2)
    ax.add_patch(neuron)
    ax.text(neuron_pos[0], neuron_pos[1], 'Σ', ha='center', va='center',
           fontsize=16, fontweight='bold')
    
    output_pos = (5, 2)
    output = Circle(output_pos, 0.3, color='lightcoral', ec='black', linewidth=2)
    ax.add_patch(output)
    ax.text(output_pos[0], output_pos[1], 'y', ha='center', va='center',
           fontsize=14, fontweight='bold')
    
    weights = ['w₁', 'w₂', 'w₃']
    for i, (pos, weight) in enumerate(zip(input_positions, weights)):
        arrow = FancyArrowPatch((pos[0]+0.3, pos[1]), (neuron_pos[0]-0.4, neuron_pos[1]),
                               arrowstyle='->', mutation_scale=20, linewidth=1.5, color='blue')
        ax.add_patch(arrow)
        mid_x = (pos[0] + neuron_pos[0]) / 2
        mid_y = pos[1] + 0.2
        ax.text(mid_x, mid_y, weight, fontsize=10, ha='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    arrow = FancyArrowPatch((neuron_pos[0]+0.4, neuron_pos[1]), (output_pos[0]-0.3, output_pos[1]),
                           arrowstyle='->', mutation_scale=20, linewidth=2, color='red')
    ax.add_patch(arrow)
    
    ax.text(3, 0.3, 'y = f(w₁x₁ + w₂x₂ + w₃x₃ + b)', ha='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 4)
    ax.set_title('Single Perceptron', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def visualize_perceptron_decision_boundary(save_path: Optional[str] = None):
    """Show how a perceptron creates a decision boundary"""
    np.random.seed(42)
    class0 = np.random.randn(50, 2) + [1, 1]
    class1 = np.random.randn(50, 2) + [3, 3]
    
    w1, w2, b = -1, 1, 0.5
    x1_boundary = np.linspace(-1, 5, 100)
    x2_boundary = -(w1 * x1_boundary + b) / w2
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(class0[:, 0], class0[:, 1], c='blue', s=100, alpha=0.6, 
              label='Class 0', edgecolors='black', linewidth=1)
    ax.scatter(class1[:, 0], class1[:, 1], c='red', s=100, alpha=0.6,
              label='Class 1', edgecolors='black', linewidth=1)
    ax.plot(x1_boundary, x2_boundary, 'g-', linewidth=3, label='Decision Boundary')
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
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def visualize_mlp_architecture(save_path: Optional[str] = None):
    """Visualize a multi-layer perceptron"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    layers = [
        [(0.5, i) for i in [4, 3, 2, 1]],
        [(2.5, i) for i in [4.5, 3.5, 2.5, 1.5]],
        [(4.5, i) for i in [4, 3, 2]],
        [(6.5, i) for i in [3.5, 2.5]]
    ]
    
    layer_labels = [
        ['x₁', 'x₂', 'x₃', 'x₄'],
        ['h₁₁', 'h₁₂', 'h₁₃', 'h₁₄'],
        ['h₂₁', 'h₂₂', 'h₂₃'],
        ['ŷ₁', 'ŷ₂']
    ]
    
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    
    for layer_idx, (positions, labels, color) in enumerate(zip(layers, layer_labels, colors)):
        for pos, label in zip(positions, labels):
            circle = Circle(pos, 0.25, color=color, ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], label, ha='center', va='center',
                   fontsize=10, fontweight='bold')
        
        if layer_idx < len(layers) - 1:
            for pos1 in positions:
                for pos2 in layers[layer_idx + 1]:
                    ax.plot([pos1[0]+0.25, pos2[0]-0.25], [pos1[1], pos2[1]],
                           'gray', linewidth=0.5, alpha=0.3)
    
    layer_names = ['Input\nLayer', 'Hidden\nLayer 1', 'Hidden\nLayer 2', 'Output\nLayer']
    for i, (name, x_pos) in enumerate(zip(layer_names, [0.5, 2.5, 4.5, 6.5])):
        ax.text(x_pos, 5.5, name, ha='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(0, 6)
    ax.set_title('Multi-Layer Perceptron (MLP)', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def visualize_activation_functions(save_path: Optional[str] = None):
    """Visualize common activation functions"""
    x = np.linspace(-5, 5, 1000)
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    relu = np.maximum(0, x)
    leaky_relu = np.where(x > 0, x, 0.01 * x)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(x, sigmoid, 'b-', linewidth=2)
    axes[0, 0].set_title('Sigmoid: σ(x) = 1/(1+e⁻ˣ)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('x', fontsize=11)
    axes[0, 0].set_ylabel('σ(x)', fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[0, 0].axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    axes[0, 0].set_ylim(-0.1, 1.1)
    
    axes[0, 1].plot(x, tanh, 'g-', linewidth=2)
    axes[0, 1].set_title('Tanh: tanh(x)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('x', fontsize=11)
    axes[0, 1].set_ylabel('tanh(x)', fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[0, 1].axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    axes[0, 1].set_ylim(-1.1, 1.1)
    
    axes[1, 0].plot(x, relu, 'r-', linewidth=2)
    axes[1, 0].set_title('ReLU: max(0, x)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('x', fontsize=11)
    axes[1, 0].set_ylabel('ReLU(x)', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[1, 0].axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    
    axes[1, 1].plot(x, leaky_relu, 'orange', linewidth=2)
    axes[1, 1].set_title('Leaky ReLU: max(0.01x, x)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('x', fontsize=11)
    axes[1, 1].set_ylabel('Leaky ReLU(x)', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[1, 1].axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def visualize_activation_derivatives(save_path: Optional[str] = None):
    """Visualize derivatives of activation functions"""
    x = np.linspace(-5, 5, 1000)
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
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def visualize_training_process(loss_history: List[float], val_loss_history: Optional[List[float]] = None,
                              save_path: Optional[str] = None):
    """Visualize training process"""
    epochs = np.arange(1, len(loss_history) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(epochs, loss_history, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    if val_loss_history:
        axes[0].plot(epochs, val_loss_history, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    if val_loss_history:
        train_acc = 1 - np.array(loss_history) / max(loss_history)
        val_acc = 1 - np.array(val_loss_history) / max(val_loss_history)
        axes[1].plot(epochs, train_acc, 'b-', linewidth=2, label='Training Accuracy', alpha=0.8)
        axes[1].plot(epochs, val_acc, 'r-', linewidth=2, label='Validation Accuracy', alpha=0.8)
        axes[1].set_ylim(0, 1.1)
    else:
        axes[1].plot(epochs, loss_history, 'b-o', linewidth=2, markersize=4)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy' if val_loss_history else 'Loss', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy' if val_loss_history else 'Training Loss', 
                     fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def visualize_backpropagation(save_path: Optional[str] = None):
    """Visualize the backpropagation process"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    forward_nodes = {
        'x': (1, 4),
        'h1': (3, 4),
        'h2': (3, 2),
        'y': (5, 3),
        'L': (7, 3)
    }
    
    for name, (x, y) in forward_nodes.items():
        if name == 'L':
            rect = Rectangle((x-0.3, y-0.2), 0.6, 0.4, color='lightcoral', ec='black', linewidth=2)
            ax.add_patch(rect)
        else:
            circle = Circle((x, y), 0.25, color='lightblue', ec='black', linewidth=2)
            ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center', fontsize=12, fontweight='bold')
    
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
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


if __name__ == "__main__":
    print("=== XOR Problem ===")
    xor_problem_example()
    
    print("\n=== Linear Regression ===")
    linear_regression_example()
    
    print("\n=== Generating Visualizations ===")
    print("Run individual visualization functions to see graphs!")