"""
Logistic Regression with Sigmoid Activation

This is the bridge from Perceptron to Neural Networks!

Key Differences from Perceptron:
- Uses Sigmoid activation (differentiable) instead of Step function
- Outputs probabilities (0-1) instead of hard decisions (0/1)
- Uses Gradient Descent for training (not perceptron update rule)
- Uses Log Loss / Binary Cross-Entropy loss function

Core Formula:
    ŷ = σ(w·x + b)
    σ(z) = 1 / (1 + e^(-z))

Where:
    - σ is the sigmoid function (smooth, differentiable)
    - w are weights
    - b is bias
    - Output is a probability between 0 and 1
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
    y = np.where((exam_scores >= 60) & (attendance >= 60), 1, 0).astype(int)
    return X, y

# Default to 1000 samples
NUM_SAMPLES = 1000
X, y = generate_data(num_samples=NUM_SAMPLES, seed=0)

# =========================================================
# 2) Sigmoid Activation Function
#    This is the KEY difference from Perceptron!
#    Perceptron uses: step(z) = {1 if z >= 0, else 0}  (NOT differentiable)
#    Logistic Regression uses: σ(z) = 1/(1+e^(-z))     (DIFFERENTIABLE!)
# =========================================================
def sigmoid(z):
    """
    Sigmoid activation function.
    Input: z (can be scalar or array)
    Output: probability between 0 and 1
    
    Properties:
    - Smooth and differentiable everywhere
    - Output range: (0, 1)
    - S-shaped curve
    """
    # Clip z to prevent overflow
    z = np.clip(z, -250, 250)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """
    Derivative of sigmoid function.
    σ'(z) = σ(z) * (1 - σ(z))
    
    This is needed for gradient descent!
    """
    s = sigmoid(z)
    return s * (1 - s)

# =========================================================
# 3) Prediction Function
#    For one sample:    z = w·x + b
#    Output:            ŷ = σ(z)  (probability, not hard 0/1!)
#    For many samples:  vectorized prediction
# =========================================================
def predict_one(x, w, b):
    """
    Predict probability for a single input vector x.
    Returns: probability between 0 and 1
    """
    z = np.dot(w, x) + b
    return sigmoid(z)

def predict(X, w, b):
    """
    Predict probabilities for all samples in matrix X.
    Returns: array of probabilities
    """
    z = np.dot(X, w) + b
    return sigmoid(z)

def predict_class(X, w, b, threshold=0.5):
    """
    Convert probabilities to binary predictions (0 or 1).
    threshold: decision boundary (default 0.5)
    """
    probabilities = predict(X, w, b)
    return (probabilities >= threshold).astype(int)

# =========================================================
# 4) Loss Function: Binary Cross-Entropy (Log Loss)
#    This measures how wrong our predictions are.
#    L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
# =========================================================
def binary_cross_entropy_loss(y_true, y_pred):
    """
    Binary Cross-Entropy Loss (Log Loss).
    
    L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
    
    Properties:
    - Penalizes confident wrong predictions heavily
    - Works well with probabilities
    - Differentiable everywhere
    """
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)

# =========================================================
# 5) Gradient Computation
#    Compute gradients of loss with respect to weights and bias.
#    This is where calculus comes in!
# =========================================================
def compute_gradients(X, y, w, b):
    """
    Compute gradients using calculus (chain rule).
    
    Forward pass:
        z = X·w + b
        ŷ = σ(z)
        L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
    
    Backward pass (gradients):
        ∂L/∂ŷ = -(y/ŷ - (1-y)/(1-ŷ))
        ∂ŷ/∂z = σ'(z) = σ(z)·(1-σ(z))
        ∂z/∂w = X
        ∂z/∂b = 1
        
    Using chain rule:
        ∂L/∂w = (∂L/∂ŷ) · (∂ŷ/∂z) · (∂z/∂w) = (ŷ - y) · X
        ∂L/∂b = (∂L/∂ŷ) · (∂ŷ/∂z) · (∂z/∂b) = (ŷ - y)
    """
    m = X.shape[0]  # number of samples
    
    # Forward pass
    z = np.dot(X, w) + b
    y_pred = sigmoid(z)
    
    # Compute gradients
    error = y_pred - y  # This is (ŷ - y) after simplification
    dw = (1/m) * np.dot(X.T, error)
    db = (1/m) * np.sum(error)
    
    return dw, db

# =========================================================
# 6) Gradient Descent Training
#    This is the learning algorithm!
#    Update rule: w = w - learning_rate * gradient
# =========================================================
def train_logistic_regression(X, y, learning_rate=0.01, epochs=1000, verbose=True):
    """
    Train logistic regression using gradient descent.
    
    Returns:
        w: learned weights
        b: learned bias
        loss_history: list of loss values during training
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
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Print progress
        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
    
    return w, b, loss_history

# =========================================================
# 7) Actual Training
# =========================================================
print("=" * 70)
print("Logistic Regression Training")
print("=" * 70)
print(f"\nDataset: {NUM_SAMPLES} samples")
print(f"Features: Exam Score, Attendance Percentage")
print(f"Target: Pass (1) or Fail (0)")

w, b, loss_history = train_logistic_regression(X, y, learning_rate=0.01, epochs=1000)

print(f"\nFinal learned weights w = {w}")
print(f"Final learned bias b = {b:.6f}")
print(f"Final loss: {loss_history[-1]:.6f}")

# =========================================================
# 8) Plot Decision Boundary
#    Show the probability surface and decision boundary
# =========================================================
def plot_decision_boundary(X, y, w, b):
    """
    Visualize 2D data with logistic regression decision boundary.
    Shows probability contours and decision boundary at p=0.5
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
    Z = predict(grid_points, w, b)
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
    ax1.set_title("Logistic Regression: Decision Boundary & Probability Surface", 
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
    ax2.set_title("3D Probability Surface", fontsize=12, fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('docs/images/logistic_regression_boundary.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_decision_boundary(X, y, w, b)

# =========================================================
# 9) Plot Training Loss
#    See how loss decreases during training
# =========================================================
plt.figure(figsize=(10, 6))
plt.plot(loss_history, linewidth=2, color='blue')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Binary Cross-Entropy Loss", fontsize=12)
plt.title("Logistic Regression Training: Loss Evolution", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Log scale to see convergence better
plt.tight_layout()
plt.savefig('docs/images/logistic_regression_loss.png', dpi=150, bbox_inches='tight')
plt.show()

# =========================================================
# 10) Compare Perceptron vs Logistic Regression
#     Show the difference: hard decisions vs probabilities
# =========================================================
def compare_perceptron_vs_logistic():
    """
    Visualize the key difference: step function vs sigmoid
    """
    z = np.linspace(-5, 5, 1000)
    
    # Step function (Perceptron)
    step = np.where(z >= 0, 1, 0)
    
    # Sigmoid function (Logistic Regression)
    sig = sigmoid(z)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Step vs Sigmoid
    axes[0].plot(z, step, 'r-', linewidth=3, label='Step Function (Perceptron)', alpha=0.7)
    axes[0].plot(z, sig, 'b-', linewidth=3, label='Sigmoid (Logistic Regression)', alpha=0.7)
    axes[0].axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].set_xlabel('z = w·x + b', fontsize=12)
    axes[0].set_ylabel('Output', fontsize=12)
    axes[0].set_title('Step vs Sigmoid Activation', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.1, 1.1)
    
    # Right: Derivatives
    step_deriv = np.zeros_like(z)  # Step function is not differentiable at 0
    sig_deriv = sigmoid_derivative(z)
    
    axes[1].plot(z, step_deriv, 'r-', linewidth=2, label='Step Derivative (undefined at 0)', 
                alpha=0.7, linestyle='--')
    axes[1].plot(z, sig_deriv, 'b-', linewidth=3, label='Sigmoid Derivative', alpha=0.7)
    axes[1].axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].set_xlabel('z = w·x + b', fontsize=12)
    axes[1].set_ylabel("Derivative", fontsize=12)
    axes[1].set_title('Why Sigmoid? It\'s Differentiable!', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('docs/images/step_vs_sigmoid.png', dpi=150, bbox_inches='tight')
    plt.show()

compare_perceptron_vs_logistic()

# =========================================================
# 11) Predict for New Students (with probabilities!)
# =========================================================
print("\n" + "=" * 70)
print("Predictions on New Students (with Probabilities!)")
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
    prob = predict_one(student, w, b)
    pred_class = 1 if prob >= 0.5 else 0
    
    print(f"\nStudent {i}:")
    print(f"  Exam Score: {student[0]:.1f}, Attendance: {student[1]:.1f}%")
    print(f"  Probability of Pass: {prob:.4f} ({prob*100:.2f}%)")
    print(f"  Prediction: {'Pass ✅' if pred_class == 1 else 'Fail ❌'}")
    print(f"  Confidence: {'High' if abs(prob - 0.5) > 0.3 else 'Medium' if abs(prob - 0.5) > 0.1 else 'Low'}")

# =========================================================
# 12) User Input Test Section
# =========================================================
def test_with_user_input():
    print("\n" + "=" * 70)
    print("Test Logistic Regression with Your Own Input")
    print("=" * 70)
    try:
        user_exam = float(input("Enter exam score (e.g., 60): ").strip())
        user_attendance = float(input("Enter attendance percentage (e.g., 70): ").strip())
        user_sample = np.array([user_exam, user_attendance], dtype=float)
        
        prob = predict_one(user_sample, w, b)
        pred_class = 1 if prob >= 0.5 else 0
        
        print(f"\nYour input: Exam Score = {user_exam}, Attendance = {user_attendance}%")
        print(f"Probability of Pass: {prob:.4f} ({prob*100:.2f}%)")
        if pred_class == 1:
            print("Result: Pass ✅ (Logistic Regression predicts this student WILL pass)")
        else:
            print("Result: Fail ❌ (Logistic Regression predicts this student will NOT pass)")
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
print("1. ✅ Sigmoid is DIFFERENTIABLE → enables Gradient Descent")
print("2. ✅ Outputs PROBABILITIES (0-1) → not just hard 0/1")
print("3. ✅ Uses BINARY CROSS-ENTROPY LOSS → proper loss function")
print("4. ✅ GRADIENT DESCENT training → uses calculus!")
print("5. ✅ This is the bridge from Perceptron to Neural Networks!")
print("=" * 70)
