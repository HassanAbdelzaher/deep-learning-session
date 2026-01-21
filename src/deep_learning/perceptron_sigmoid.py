"""
Perceptron with Sigmoid Activation

This is similar to the basic perceptron, but uses Sigmoid activation instead of Step function.

Key Differences from Basic Perceptron:
- Uses Sigmoid activation (differentiable) instead of Step function
- Outputs probabilities (0-1) instead of hard decisions (0/1)
- Uses Gradient Descent for training (not perceptron update rule)
- Uses Binary Cross-Entropy loss function

This bridges the gap between Perceptron and Logistic Regression!
"""

import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1) Generate Synthetic Training Data (parameterized number of students)
#    Features: [Exam Score, Attendance Percentage]
#    Labels: y = 1 if student would "pass", else 0 (simple rule)
# =========================================================
def generate_data(num_samples=100, seed=0):
    np.random.seed(seed)  # For reproducibility

    # Generate exam scores between 40 and 100
    exam_scores = np.random.uniform(40, 100, num_samples)
    # Generate attendance between 40% and 100%
    attendance = np.random.uniform(40, 100, num_samples)
    X = np.column_stack([exam_scores, attendance])

    # Simple rule for passing: pass if exam score >= 60 OR attendance >= 60
    y = np.where((exam_scores >= 60) | (attendance >= 60), 1, 0).astype(int)
    return X, y

# Default to 1000 samples if not specified elsewhere
NUM_SAMPLES = 100
X, y = generate_data(num_samples=NUM_SAMPLES, seed=0)

# =========================================================
# 2) Activation Function (Sigmoid Function)
#    The perceptron uses a sigmoid (smooth) activation: outputs probability between 0 and 1.
#    This is DIFFERENTIABLE, unlike the step function!
# =========================================================
def sigmoid(z):
    """
    Sigmoid activation function for perceptron.
    Input:
        z (float or array): summed input (w.x + b)
    Output:
        float or array: probability between 0 and 1
    """
    # Clip z to prevent overflow
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
# 3) Prediction Function
#    For one sample:    z = w.x + b
#    Output:            y_hat = sigmoid(z)  (probability!)
#    For many samples:  vectorized prediction
# =========================================================
def predict_one(x: np.ndarray, w: np.ndarray, b: float) -> float:
    """
    Predict probability (0-1) for a single input vector x using weights w, bias b
    Returns: probability (float between 0 and 1)
    """
    z = np.dot(w, x) + b
    return sigmoid(z)

def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """
    Predict probabilities (0-1) for all samples in matrix X
    Returns: array of probabilities
    """
    z = np.dot(X, w) + b
    return sigmoid(z)

def predict_class(X: np.ndarray, w: np.ndarray, b: float, threshold: float = 0.5) -> np.ndarray:
    """
    Convert probabilities to binary predictions (0 or 1)
    threshold: decision boundary (default 0.5)
    """
    probabilities = predict(X, w, b)
    return (probabilities >= threshold).astype(int)

# =========================================================
# 4) Loss Function: Binary Cross-Entropy
#    This measures how wrong our predictions are.
#    L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
# =========================================================
def binary_cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Binary Cross-Entropy Loss (Log Loss).
    L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
    """
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)

# =========================================================
# 5) Gradient Computation
#    Compute gradients using calculus (chain rule)
# =========================================================
def compute_gradients(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float):
    """
    Compute gradients of loss with respect to weights and bias.
    
    Using chain rule:
        ∂L/∂w = (1/m) * X^T · (ŷ - y)
        ∂L/∂b = (1/m) * sum(ŷ - y)
    """
    m = X.shape[0]  # number of samples
    
    # Forward pass
    z = np.dot(X, w) + b
    y_pred = sigmoid(z)
    
    # Compute gradients
    error = y_pred - y
    dw = (1/m) * np.dot(X.T, error)
    db = (1/m) * np.sum(error)
    
    return dw, db

# =========================================================
# 6) Perceptron Training with Gradient Descent
#    Unlike basic perceptron, we use gradient descent!
#    Update rule: w = w - learning_rate * gradient
# =========================================================
def train_perceptron_sigmoid(X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000):
    """
    Trains a perceptron with sigmoid activation using gradient descent.
    Returns:
        w: learned weights (shape: n_features)
        b: learned bias (float)
        loss_history: list of loss values per epoch
    """
    # Initialize weights and bias (small random values)
    np.random.seed(42)
    w = np.random.randn(X.shape[1]) * 0.01
    b = 0.0
    
    loss_history = []
    
    for epoch in range(epochs):
        # Forward pass: compute predictions
        y_pred = predict(X, w, b)
        
        # Compute loss
        loss = binary_cross_entropy_loss(y, y_pred)
        loss_history.append(loss)
        
        # Compute gradients
        dw, db = compute_gradients(X, y, w, b)
        
        # Update weights and bias using gradient descent
        w = w - lr * dw
        b = b - lr * db
        
        # Print progress
        if (epoch + 1) % 100 == 0:
            # Also compute accuracy
            y_pred_class = predict_class(X, w, b)
            accuracy = np.mean(y_pred_class == y)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}, Accuracy: {accuracy:.4f}")
    
    return w, b, loss_history

# =========================================================
# 7) Actual Training
#    Train the perceptron using gradient descent.
# =========================================================
w, b, loss_history = train_perceptron_sigmoid(X, y, lr=0.01, epochs=1000)

print("Final learned weights w =", w)
print("Final learned bias b =", b)
print(f"Final loss: {loss_history[-1]:.6f}")

# Compute final accuracy
y_pred_final = predict_class(X, w, b)
final_accuracy = np.mean(y_pred_final == y)
print(f"Final accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")

# =========================================================
# 8) Plot Data + Decision Boundary
#    Show the input samples, label colors, and the decision boundary found.
#    - Decision line: w1 * x1 + w2 * x2 + b = 0 (where probability = 0.5)
# =========================================================
def plot_decision_boundary(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float):
    """
    Visualize 2D data with the perceptron decision boundary (probability surface).
    """
    passed = X[y == 1]
    failed = X[y == 0]
    
    plt.figure(figsize=(12, 8))
    
    # Plot data points
    plt.scatter(passed[:, 0], passed[:, 1], color="green", label="Pass (1)", 
                alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    plt.scatter(failed[:, 0], failed[:, 1], color="red", label="Fail (0)", 
                alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Create grid for probability surface
    x1_min, x1_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    x2_min, x2_max = X[:, 1].min() - 5, X[:, 1].max() + 5
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                           np.linspace(x2_min, x2_max, 100))
    
    # Compute probabilities for grid
    grid_points = np.c_[xx1.ravel(), xx2.ravel()]
    Z = predict(grid_points, w, b)
    Z = Z.reshape(xx1.shape)
    
    # Plot probability contours
    contour = plt.contourf(xx1, xx2, Z, levels=20, cmap='RdYlGn', alpha=0.3)
    plt.colorbar(contour, label='Probability of Pass')
    
    # Decision boundary (where probability = 0.5)
    contour_line = plt.contour(xx1, xx2, Z, levels=[0.5], colors='black', 
                              linewidths=2, linestyles='--')
    plt.clabel(contour_line, inline=True, fontsize=10, fmt='Decision Boundary (p=0.5)')
    
    # Labels and legend
    plt.xlabel("Exam Score (x1)", fontsize=12)
    plt.ylabel("Attendance Percentage (x2)", fontsize=12)
    plt.title("Perceptron with Sigmoid: Data Points & Learned Decision Boundary", 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('docs/images/perceptron_sigmoid_boundary.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_decision_boundary(X, y, w, b)

# =========================================================
# 9) Plot Loss Evolution During Training
#    See how loss decreases with epochs using gradient descent.
# =========================================================
plt.figure(figsize=(10, 6))
plt.plot(loss_history, marker='o', markersize=3, linewidth=2, color='blue')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Binary Cross-Entropy Loss", fontsize=12)
plt.title("Perceptron with Sigmoid Training: Loss Evolution per Epoch", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Log scale to see convergence better
plt.tight_layout()
plt.savefig('docs/images/perceptron_sigmoid_loss.png', dpi=150, bbox_inches='tight')
plt.show()

# =========================================================
# 10) Predict for a New Student (with probability!)
#     Example: Given a new student's exam score and attendance, 
#     what's the probability they will pass?
# =========================================================
new_student = np.array([72, 68], dtype=float)  # Example: exam score 72, attendance 68
prob = predict_one(new_student, w, b)
pred_class = 1 if prob >= 0.5 else 0

print("\nPrediction for new student: Exam Score =", new_student[0], "Attendance =", new_student[1])
print(f"Probability of Pass: {prob:.4f} ({prob*100:.2f}%)")
if pred_class == 1:
    print("Result: Pass ✅ (The perceptron predicts this student WILL pass)")
else:
    print("Result: Fail ❌ (The perceptron predicts this student will NOT pass)")

# =========================================================
# 11) User Input Test Section
#     Allow user to enter new student data and display perceptron prediction with probability.
# =========================================================

def test_with_user_input():
    print("\n=== Test the perceptron with sigmoid using your own input ===")
    try:
        user_exam = float(input("Enter exam score (e.g., 60): ").strip())
        user_attendance = float(input("Enter attendance percentage (e.g., 70): ").strip())
        user_sample = np.array([user_exam, user_attendance], dtype=float)
        user_prob = predict_one(user_sample, w, b)
        user_pred = 1 if user_prob >= 0.5 else 0
        
        print(f"\nYour input: Exam Score = {user_exam}, Attendance = {user_attendance}")
        print(f"Probability of Pass: {user_prob:.4f} ({user_prob*100:.2f}%)")
        if user_pred == 1:
            print("Result: Pass ✅ (The perceptron predicts this student WILL pass)")
        else:
            print("Result: Fail ❌ (The perceptron predicts this student will NOT pass)")
    except Exception as e:
        print(f"Error with your input: {e}")
        print("Please make sure to enter valid numerical values.")

# Uncomment this to enable user input test when running interactively
if __name__ == "__main__":
    # Demo using user input (comment out if running in non-interactive mode)
    # test_with_user_input()
    pass

print("\n" + "=" * 70)
print("Key Differences from Basic Perceptron:")
print("=" * 70)
print("1. ✅ Uses SIGMOID activation (differentiable) instead of Step function")
print("2. ✅ Outputs PROBABILITIES (0-1) instead of hard 0/1")
print("3. ✅ Uses GRADIENT DESCENT for training (not perceptron update rule)")
print("4. ✅ Uses BINARY CROSS-ENTROPY LOSS (proper loss function)")
print("5. ✅ This bridges Perceptron to Neural Networks!")
print("=" * 70)
