"""
Multi-Layer Perceptron (MLP)

This extends the single perceptron to multiple layers, enabling learning of complex patterns!

Key Features:
- Multiple layers: Input -> Hidden -> Output
- Sigmoid activation functions
- Backpropagation algorithm
- Gradient Descent training
- Can learn non-linear decision boundaries

Architecture:
    Input Layer (2 neurons) -> Hidden Layer (4 neurons) -> Output Layer (1 neuron)

This is the foundation of all deep neural networks!
"""

import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1) Generate Synthetic Training Data
#    Features: [Exam Score, Attendance Percentage]
#    Labels: y = 1 if student would "pass", else 0
# =========================================================
def generate_data(num_samples=1000, seed=0):
    np.random.seed(seed)  # For reproducibility

    # Generate exam scores between 40 and 100
    exam_scores = np.random.uniform(40, 100, num_samples)
    # Generate attendance between 40% and 100%
    attendance = np.random.uniform(40, 100, num_samples)
    X = np.column_stack([exam_scores, attendance])

    # Simple rule for passing: pass if exam score >= 60 OR attendance >= 60
    y = np.where((exam_scores >= 60) | (attendance >= 60), 1, 0).astype(int)
    return X, y

# Default to 1000 samples
NUM_SAMPLES = 1000
X, y = generate_data(num_samples=NUM_SAMPLES, seed=0)

# =========================================================
# 2) Activation Functions
#    Sigmoid for smooth, differentiable activation
# =========================================================
def sigmoid(z):
    """
    Sigmoid activation function.
    Input: z (can be scalar or array)
    Output: probability between 0 and 1
    """
    z = np.clip(z, -250, 250)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """
    Derivative of sigmoid function.
    σ'(z) = σ(z) * (1 - σ(z))
    """
    s = sigmoid(z)
    return s * (1 - s)

# =========================================================
# 3) Multi-Layer Perceptron Class
#    This implements a neural network with multiple layers
# =========================================================
class MultiLayerPerceptron:
    """
    Multi-Layer Perceptron (MLP) with configurable layers.
    
    Architecture: [input_size, hidden_size, ..., output_size]
    Example: [2, 4, 1] means:
        - Input: 2 features
        - Hidden: 4 neurons
        - Output: 1 neuron (binary classification)
    """
    
    def __init__(self, layers=[2, 4, 1], learning_rate=0.01):
        """
        Initialize MLP.
        
        Args:
            layers: List of layer sizes [input, hidden1, hidden2, ..., output]
            learning_rate: Learning rate for gradient descent
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases for each layer
        np.random.seed(42)
        for i in range(len(layers) - 1):
            # Xavier initialization
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Returns:
            activations: List of activations for each layer
            z_values: List of pre-activation values (for backprop)
        """
        activations = [X]  # Input layer
        z_values = []
        
        # Propagate through each layer
        for i in range(len(self.weights)):
            # Compute weighted sum: z = X·W + b
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            # Apply activation function (sigmoid)
            a = sigmoid(z)
            activations.append(a)
        
        return activations, z_values
    
    def backward(self, X, y, activations, z_values):
        """
        Backward propagation (backpropagation algorithm).
        
        Computes gradients using chain rule:
        1. Compute output error
        2. Propagate error backward through layers
        3. Compute gradients for weights and biases
        
        Returns:
            weight_gradients: List of gradients for weights
            bias_gradients: List of gradients for biases
        """
        m = X.shape[0]  # number of samples
        num_layers = len(self.weights)
        
        # Initialize gradients
        weight_gradients = [np.zeros_like(w) for w in self.weights]
        bias_gradients = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error (derivative of loss with respect to output)
        # For binary cross-entropy: error = (prediction - target)
        output_error = activations[-1] - y
        
        # Backpropagate through layers (from output to input)
        delta = output_error
        for i in range(num_layers - 1, -1, -1):
            # Compute gradients for this layer
            weight_gradients[i] = (1/m) * np.dot(activations[i].T, delta)
            bias_gradients[i] = (1/m) * np.sum(delta, axis=0, keepdims=True)
            
            # Propagate error to previous layer (if not input layer)
            if i > 0:
                # Error from this layer to previous layer
                delta = np.dot(delta, self.weights[i].T)
                # Apply derivative of activation function
                delta *= sigmoid_derivative(z_values[i-1])
        
        return weight_gradients, bias_gradients
    
    def update_weights(self, weight_gradients, bias_gradients):
        """Update weights and biases using gradient descent"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
    
    def predict(self, X):
        """Make predictions (returns probabilities)"""
        activations, _ = self.forward(X)
        return activations[-1]
    
    def predict_class(self, X, threshold=0.5):
        """Convert probabilities to binary predictions"""
        probabilities = self.predict(X)
        return (probabilities >= threshold).astype(int)
    
    def train(self, X, y, epochs=1000, verbose=True):
        """
        Train the MLP using gradient descent.
        
        Returns:
            loss_history: List of loss values during training
        """
        loss_history = []
        
        for epoch in range(epochs):
            # Forward pass
            activations, z_values = self.forward(X)
            
            # Compute loss (binary cross-entropy)
            y_pred = activations[-1]
            y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
            loss = -np.mean(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
            loss_history.append(loss)
            
            # Backward pass (backpropagation)
            weight_gradients, bias_gradients = self.backward(X, y, activations, z_values)
            
            # Update weights
            self.update_weights(weight_gradients, bias_gradients)
            
            # Print progress
            if verbose and (epoch + 1) % 100 == 0:
                y_pred_class = self.predict_class(X)
                accuracy = np.mean(y_pred_class == y)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}, Accuracy: {accuracy:.4f}")
        
        return loss_history

# =========================================================
# 4) Create and Train MLP
#    Architecture: [2 inputs, 4 hidden neurons, 1 output]
# =========================================================
print("=" * 70)
print("Multi-Layer Perceptron Training")
print("=" * 70)
print(f"\nDataset: {NUM_SAMPLES} samples")
print(f"Features: Exam Score, Attendance Percentage")
print(f"Target: Pass (1) or Fail (0)")
print(f"\nArchitecture: [2 inputs -> 4 hidden neurons -> 1 output]")

# Create MLP
mlp = MultiLayerPerceptron(layers=[2, 4, 1], learning_rate=0.01)

# Train the model
print("\nTraining...")
loss_history = mlp.train(X, y, epochs=1000)

# Final evaluation
y_pred_final = mlp.predict_class(X)
final_accuracy = np.mean(y_pred_final == y)
final_loss = loss_history[-1]

print(f"\nFinal Loss: {final_loss:.6f}")
print(f"Final Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")

# Print learned weights (for understanding)
print("\nLearned Weights:")
for i, (w, b) in enumerate(zip(mlp.weights, mlp.biases)):
    layer_name = "Input->Hidden" if i == 0 else "Hidden->Output"
    print(f"  {layer_name} Layer:")
    print(f"    Weights shape: {w.shape}")
    print(f"    Bias shape: {b.shape}")

# =========================================================
# 5) Plot Decision Boundary
#    Show how MLP learns complex decision boundaries
# =========================================================
def plot_decision_boundary(X, y, mlp):
    """
    Visualize 2D data with MLP decision boundary.
    Shows probability surface learned by the multi-layer network.
    """
    passed = X[y == 1]
    failed = X[y == 0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Data points with decision boundary
    ax1.scatter(passed[:, 0], passed[:, 1], color="green", label="Pass (1)", 
                alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax1.scatter(failed[:, 0], failed[:, 1], color="red", label="Fail (0)", 
                alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Create grid for probability surface
    x1_min, x1_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    x2_min, x2_max = X[:, 1].min() - 5, X[:, 1].max() + 5
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                           np.linspace(x2_min, x2_max, 100))
    
    # Compute probabilities for grid
    grid_points = np.c_[xx1.ravel(), xx2.ravel()]
    Z = mlp.predict(grid_points)
    Z = Z.reshape(xx1.shape)
    
    # Plot probability contours
    contour = ax1.contourf(xx1, xx2, Z, levels=20, cmap='RdYlGn', alpha=0.3)
    plt.colorbar(contour, ax=ax1, label='Probability of Pass')
    
    # Decision boundary (where probability = 0.5)
    contour_line = ax1.contour(xx1, xx2, Z, levels=[0.5], colors='black', 
                               linewidths=2, linestyles='--')
    ax1.clabel(contour_line, inline=True, fontsize=10, fmt='Decision Boundary (p=0.5)')
    
    ax1.set_xlabel("Exam Score", fontsize=12)
    ax1.set_ylabel("Attendance Percentage", fontsize=12)
    ax1.set_title("Multi-Layer Perceptron: Decision Boundary & Probability Surface", 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: 3D probability surface
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(xx1, xx2, Z, cmap='RdYlGn', alpha=0.8, 
                           linewidth=0, antialiased=True)
    ax2.scatter(passed[:, 0], passed[:, 1], y[y == 1], color='green', 
               s=20, alpha=0.6, label='Pass')
    ax2.scatter(failed[:, 0], failed[:, 1], y[y == 0], color='red', 
               s=20, alpha=0.6, label='Fail')
    ax2.set_xlabel("Exam Score", fontsize=11)
    ax2.set_ylabel("Attendance", fontsize=11)
    ax2.set_zlabel("Probability", fontsize=11)
    ax2.set_title("3D Probability Surface (Learned by MLP)", fontsize=12, fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('docs/images/multi_layer_perceptron_boundary.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_decision_boundary(X, y, mlp)

# =========================================================
# 6) Plot Training Loss
#    See how loss decreases during training
# =========================================================
plt.figure(figsize=(10, 6))
plt.plot(loss_history, linewidth=2, color='blue')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Binary Cross-Entropy Loss", fontsize=12)
plt.title("Multi-Layer Perceptron Training: Loss Evolution", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Log scale to see convergence better
plt.tight_layout()
plt.savefig('docs/images/multi_layer_perceptron_loss.png', dpi=150, bbox_inches='tight')
plt.show()

# =========================================================
# 7) Visualize Network Architecture
#    Show the structure of the MLP
# =========================================================
def visualize_network_architecture(mlp):
    """Visualize the MLP architecture"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    layer_sizes = mlp.layers
    num_layers = len(layer_sizes)
    
    # Calculate positions for each layer
    layer_x_positions = np.linspace(1, 9, num_layers)
    node_radius = 0.15
    
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    # Draw nodes and connections
    for layer_idx, (layer_size, x_pos) in enumerate(zip(layer_sizes, layer_x_positions)):
        # Calculate vertical positions for nodes in this layer
        if layer_size == 1:
            y_positions = [4]
        else:
            y_positions = np.linspace(2, 6, layer_size)
        
        # Draw nodes
        for node_idx, y_pos in enumerate(y_positions):
            color = colors[min(layer_idx, len(colors)-1)]
            circle = plt.Circle((x_pos, y_pos), node_radius, color=color, 
                              ec='black', linewidth=2)
            ax.add_patch(circle)
            
            # Label
            if layer_idx == 0:
                label = f'x{node_idx+1}'
            elif layer_idx == num_layers - 1:
                label = 'ŷ'
            else:
                label = f'h{node_idx+1}'
            ax.text(x_pos, y_pos, label, ha='center', va='center', 
                   fontsize=10, fontweight='bold')
        
        # Draw connections to next layer
        if layer_idx < num_layers - 1:
            next_layer_size = layer_sizes[layer_idx + 1]
            if next_layer_size == 1:
                next_y_positions = [4]
            else:
                next_y_positions = np.linspace(2, 6, next_layer_size)
            
            next_x_pos = layer_x_positions[layer_idx + 1]
            
            for y_pos in y_positions:
                for next_y_pos in next_y_positions:
                    ax.plot([x_pos + node_radius, next_x_pos - node_radius], 
                           [y_pos, next_y_pos], 'gray', linewidth=0.5, alpha=0.3)
    
    # Layer labels
    layer_names = ['Input\nLayer', 'Hidden\nLayer', 'Output\nLayer']
    for i, (x_pos, name) in enumerate(zip(layer_x_positions, layer_names[:num_layers])):
        ax.text(x_pos, 7, name, ha='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        ax.text(x_pos, 0.5, f'{layer_sizes[i]} neurons', ha='center', fontsize=9)
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_title('Multi-Layer Perceptron Architecture', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('docs/images/multi_layer_perceptron_architecture.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_network_architecture(mlp)

# =========================================================
# 8) Compare Single vs Multi-Layer Perceptron
#    Show why multiple layers are powerful
# =========================================================
def compare_single_vs_multi_layer():
    """
    Compare single-layer perceptron vs multi-layer perceptron.
    This demonstrates why multiple layers enable learning complex patterns.
    """
    print("\n" + "=" * 70)
    print("Single-Layer vs Multi-Layer Perceptron")
    print("=" * 70)
    print("\nKey Differences:")
    print("1. Single-Layer Perceptron:")
    print("   - Can only learn LINEAR decision boundaries")
    print("   - Limited to simple patterns")
    print("   - Cannot solve XOR problem")
    print("\n2. Multi-Layer Perceptron:")
    print("   - Can learn NON-LINEAR decision boundaries")
    print("   - Can learn complex patterns")
    print("   - Can solve XOR problem")
    print("   - Foundation of deep learning!")
    print("\nWhy Multiple Layers?")
    print("- Each layer learns increasingly complex features")
    print("- Hidden layers enable non-linear transformations")
    print("- More layers = more capacity to learn complex patterns")

compare_single_vs_multi_layer()

# =========================================================
# 9) Predict for New Students
# =========================================================
print("\n" + "=" * 70)
print("Predictions on New Students")
print("=" * 70)

new_students = np.array([
    [72, 68],  # Example: exam score 72, attendance 68
    [55, 75],  # Example: exam score 55, attendance 75
    [85, 90],  # Example: exam score 85, attendance 90
    [45, 50],  # Example: exam score 45, attendance 50
])

print("\nPredictions:")
print("-" * 70)
for i, student in enumerate(new_students, 1):
    prob = mlp.predict(student.reshape(1, -1))[0, 0]
    pred_class = 1 if prob >= 0.5 else 0
    
    print(f"\nStudent {i}:")
    print(f"  Exam Score: {student[0]:.1f}, Attendance: {student[1]:.1f}%")
    print(f"  Probability of Pass: {prob:.4f} ({prob*100:.2f}%)")
    print(f"  Prediction: {'Pass ✅' if pred_class == 1 else 'Fail ❌'}")
    print(f"  Confidence: {'High' if abs(prob - 0.5) > 0.3 else 'Medium' if abs(prob - 0.5) > 0.1 else 'Low'}")

# =========================================================
# 10) User Input Test Section
# =========================================================
def test_with_user_input():
    print("\n" + "=" * 70)
    print("Test Multi-Layer Perceptron with Your Own Input")
    print("=" * 70)
    try:
        user_exam = float(input("Enter exam score (e.g., 60): ").strip())
        user_attendance = float(input("Enter attendance percentage (e.g., 70): ").strip())
        user_sample = np.array([[user_exam, user_attendance]], dtype=float)
        
        prob = mlp.predict(user_sample)[0, 0]
        pred_class = 1 if prob >= 0.5 else 0
        
        print(f"\nYour input: Exam Score = {user_exam}, Attendance = {user_attendance}%")
        print(f"Probability of Pass: {prob:.4f} ({prob*100:.2f}%)")
        if pred_class == 1:
            print("Result: Pass ✅ (The MLP predicts this student WILL pass)")
        else:
            print("Result: Fail ❌ (The MLP predicts this student will NOT pass)")
    except Exception as e:
        print(f"Error with your input: {e}")
        print("Please make sure to enter valid numerical values.")

# Uncomment to enable user input test
if __name__ == "__main__":
    # test_with_user_input()  # Uncomment for interactive mode
    pass

print("\n" + "=" * 70)
print("Key Takeaways:")
print("=" * 70)
print("1. ✅ MULTIPLE LAYERS enable learning complex patterns")
print("2. ✅ HIDDEN LAYERS perform non-linear transformations")
print("3. ✅ BACKPROPAGATION computes gradients through all layers")
print("4. ✅ This is the foundation of DEEP NEURAL NETWORKS!")
print("5. ✅ More layers = more capacity (but also more complexity)")
print("=" * 70)
