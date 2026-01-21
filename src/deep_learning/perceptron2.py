import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1) Generate Synthetic Training Data (parameterized number of students)
#    Features: [Exam Score]
#    Labels: y = 1 if student would "pass", else 0 (based only on score)
# =========================================================
def generate_data(num_samples=100, seed=0):
    np.random.seed(seed)  # For reproducibility

    # Generate exam scores between 40 and 100
    exam_scores = np.random.uniform(40, 100, num_samples).reshape(-1, 1)

    # Simple rule for passing: pass if exam score >= 60
    y = np.where(exam_scores[:, 0] >= 60, 1, 0).astype(int)
    return exam_scores, y

# Default to 1000 samples if not specified elsewhere
NUM_SAMPLES = 1000
X, y = generate_data(num_samples=NUM_SAMPLES, seed=0)

# =========================================================
# 2) Activation Function (Step Function)
#    The perceptron uses a step (threshold) activation: outputs 1 if input >=0, else 0.
#    This implements the decision boundary.
# =========================================================
def step(z: float) -> int:
    """
    Step activation function for perceptron.
    Input:
        z (float): summed input (w.x + b)
    Output:
        int: 1 if z >= 0, else 0
    """
    return 1 if z >= 0 else 0

# =========================================================
# 3) Prediction Function
#    For one sample:    z = w.x + b
#    Output:            y_hat = step(z)
#    For many samples:  vectorized prediction
# =========================================================
def predict_one(x: np.ndarray, w: np.ndarray, b: float) -> int:
    """
    Predict (0/1) for a single input vector x using weights w, bias b
    """
    z = np.dot(w, x) + b
    return step(z)

def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """
    Predict (0/1) for all samples in matrix X
    """
    return np.array([predict_one(x, w, b) for x in X], dtype=int)

# =========================================================
# 4) Perceptron Training (Learning Algorithm)
#    Classic perceptron update rule for score-only feature
# =========================================================
def train_perceptron(X: np.ndarray, y: np.ndarray, lr: float = 0.9, epochs: int = 50):
    """
    Trains a perceptron on dataset (X, y).
    Returns:
        w: learned weights (shape: n_features)
        b: learned bias (float)
        errors_history: number of classification errors per epoch
    """
    # Now n_features = 1
    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0

    errors_history = []

    for epoch in range(epochs):
        errors = 0
        for xi, yi in zip(X, y):
            y_hat = predict_one(xi, w, b)
            error = yi - y_hat

            if error != 0:
                w = w + lr * error * xi
                b = b + lr * error
                errors += 1

        errors_history.append(errors)
        if errors == 0:
            print(f"Early stopping at epoch {epoch} (zero errors reached)")
            break

    return w, b, errors_history

# =========================================================
# 5) Actual Training
# =========================================================
w, b, errors_history = train_perceptron(X, y, lr=0.5, epochs=500)

print("Final learned weights w =", w)
print("Final learned bias b =", b)
print("Errors per epoch (last 10 epochs):", errors_history[-10:])

# =========================================================
# 6) Plot Data + Decision Boundary
#    Show the input samples, label colors, and the decision boundary found.
#    - Decision line: w1 * x1 + b = 0  -->  x1 = -b/w1 (since only one feature)
# =========================================================
def plot_decision_boundary(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float):
    """
    Visualize 1D data with the perceptron decision boundary (vertical line).
    """
    passed = X[y == 1]
    failed = X[y == 0]

    plt.figure(figsize=(8, 3))

    # Plot data points along single feature axis
    plt.scatter(passed[:, 0], np.zeros_like(passed[:, 0]) + 0.2, color="green", label="Pass (1)", alpha=0.6, s=60)
    plt.scatter(failed[:, 0], np.zeros_like(failed[:, 0]) - 0.2, color="red", label="Fail (0)", alpha=0.6, s=60)

    # Mark the decision boundary: x1 = -b/w1
    if abs(w[0]) > 1e-9:
        boundary_x = -b / w[0]
        plt.axvline(x=boundary_x, linestyle="--", color="black", label="Decision Boundary")
        plt.text(boundary_x, 0.32, f"Boundary = {boundary_x:.2f}", rotation=90)
    else:
        plt.axvline(x=0, linestyle="--", color="black", label="Degenerate Boundary")

    plt.xlabel("Exam Score")
    plt.yticks([])
    plt.ylim(-0.6, 0.6)
    plt.title("Perceptron (Score Only): Data Points & Decision Boundary")
    plt.legend()
    plt.grid(True, axis="x")
    plt.tight_layout()
    plt.show()

plot_decision_boundary(X, y, w, b)

# =========================================================
# 7) Plot Error Evolution During Training
# =========================================================
plt.figure(figsize=(8, 4))
plt.plot(errors_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Number of Classification Errors")
plt.title("Perceptron Training: Error Evolution per Epoch")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================================================
# 8) Predict for a New Student (score only)
# =========================================================
new_student_score = 72  # Example: exam score 72
new_student = np.array([new_student_score], dtype=float)
pred = predict_one(new_student, w, b)

print("\nPrediction for new student: Exam Score =", new_student_score)
if pred == 1:
    print("Result: Pass ✅ (The perceptron predicts this student WILL pass)")
else:
    print("Result: Fail ❌ (The perceptron predicts this student will NOT pass)")


# =========================================================
# 9) User Input Test Section (score only)
# =========================================================

def test_with_user_input():
    print("\n=== Test the perceptron with your own input (Score only) ===")
    try:
        user_exam = float(input("Enter exam score (e.g., 60): ").strip())
        user_sample = np.array([user_exam], dtype=float)
        user_pred = predict_one(user_sample, w, b)
        print(f"\nYour input: Exam Score = {user_exam}")
        if user_pred == 1:
            print("Result: Pass ✅ (The perceptron predicts this student WILL pass)")
        else:
            print("Result: Fail ❌ (The perceptron predicts this student will NOT pass)")
    except Exception as e:
        print(f"Error with your input: {e}")
        print("Please make sure to enter a valid numerical value.")

# Uncomment this to enable user input test when running interactively
if __name__ == "__main__":
    # Demo using user input (comment out if running in non-interactive mode)
    test_with_user_input()