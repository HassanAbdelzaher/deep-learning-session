"""
Neural Networks Fundamentals
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable


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


if __name__ == "__main__":
    print("=== XOR Problem ===")
    xor_problem_example()
    
    print("\n=== Linear Regression ===")
    linear_regression_example()
