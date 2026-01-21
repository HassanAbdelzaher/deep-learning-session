import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1) Simple Training Data (Exam Score, Attendance Percentage)
#    Each data point is a student: [Exam Score, Attendance %]
#    y: 0 = Fail, 1 = Pass
#    We want the perceptron to learn to separate fail/pass
# =========================================================
# Data: 8 students, each row: [Exam Score, Attendance Percentage]
X = np.array([
    [40, 50],  # Low score, low attendance -- expected to fail
    [50, 45],
    [55, 52],
    [60, 60],
    [65, 70],  # Higher scores, higher attendance -- expected to pass
    [70, 65],
    [80, 75],
    [85, 90]
], dtype=float)

# Labels: 0 means fail, 1 means pass, based on reasonable score/attendance
y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)

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
#    Classic perceptron update rule:
#    If the perceptron misclassifies a sample, we move the weights towards correct classification:
#        w = w + lr * (y - y_hat) * x
#        b = b + lr * (y - y_hat)
#    - lr: learning rate (how "big" the step is)
#    - epochs: number of passes over the dataset
#    Stop early if all samples classified correctly in an epoch.
#    Track errors to see learning progress.
# =========================================================
def train_perceptron(X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 50):
    """
    Trains a perceptron on dataset (X, y).
    Returns:
        w: learned weights (shape: n_features)
        b: learned bias (float)
        errors_history: number of classification errors per epoch
    """
    # Initialize weights (zeros) and bias
    w = np.zeros(X.shape[1], dtype=float)  # n_features = 2 in this case
    b = 0.0

    # Track the number of misclassifications per epoch (for plotting)
    errors_history = []

    for epoch in range(epochs):
        errors = 0  # reset counter for this pass

        # Go through every training sample
        for xi, yi in zip(X, y):
            y_hat = predict_one(xi, w, b)          # model prediction
            error = yi - y_hat                     # +1 if under-predicted, -1 if over-predicted, 0 if correct

            # Only update if there's a misclassification
            if error != 0:
                w = w + lr * error * xi            # shift weights toward correct class
                b = b + lr * error                 # update bias
                errors += 1                        # count this mistake

        errors_history.append(errors)

        # If no errors, we've learned a perfect separator
        if errors == 0:
            print(f"Early stopping at epoch {epoch} (zero errors reached)")
            break

    return w, b, errors_history

# =========================================================
# 5) Actual Training
#    Train the perceptron using the above logic.
#    Adjust learning_rate (lr) or epochs if training doesn't converge.
#    Print the learned weights, bias, and last errors for diagnostics.
# =========================================================
w, b, errors_history = train_perceptron(X, y, lr=0.01, epochs=200)

print("Final learned weights w =", w)
print("Final learned bias b =", b)
print("Errors per epoch (last 10 epochs):", errors_history[-10:])  # Can indicate convergence

# =========================================================
# 6) Plot Data + Decision Boundary
#    Show the input samples, label colors, and the decision boundary found.
#    - Decision line: w1 * x1 + w2 * x2 + b = 0
#    - Can be solved for x2: x2 = -(w1*x1 + b) / w2 (unless w2 ~ 0, then vertical)
# =========================================================
def plot_decision_boundary(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float):
    """
    Visualize 2D data with the perceptron decision boundary.
    """
    passed = X[y == 1]
    failed = X[y == 0]

    plt.figure(figsize=(8, 6))

    # Plot data points
    plt.scatter(passed[:, 0], passed[:, 1], color="green", label="Pass (1)")
    plt.scatter(failed[:, 0], failed[:, 1], color="red", label="Fail (0)")

    # Range for exam scores (x1 axis)
    x1_min, x1_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    x1_values = np.linspace(x1_min, x1_max, 200)

    # Decision boundary
    if abs(w[1]) < 1e-9:
        # w2 nearly zero: vertical line at x1 = -b/w1
        x1_const = -b / w[0] if abs(w[0]) > 1e-9 else 0
        plt.axvline(x=x1_const, linestyle="--", color="black", label="Decision Boundary")
        plt.text(x1_const, plt.ylim()[1]-5, f"Boundary (vertical)", rotation=90)
    else:
        # Regular case: plot x2 as function of x1
        x2_values = -(w[0] * x1_values + b) / w[1]
        plt.plot(x1_values, x2_values, linestyle="--", color="blue", label="Decision Boundary")

    # Labels and legend
    plt.xlabel("Exam Score (x1)")
    plt.ylabel("Attendance Percentage (x2)")
    plt.title("Perceptron: Data Points & Learned Decision Boundary")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_decision_boundary(X, y, w, b)

# =========================================================
# 7) Plot Error Evolution During Training
#    See how errors decrease with epochs. When it hits zero, model fits perfectly.
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
# 8) Predict for a New Student
#    Example: Given a new student's exam score and attendance, will they pass?
#    The perceptron prediction is shown, with a user-friendly label.
# =========================================================
new_student = np.array([72, 68], dtype=float)  # Example: exam score 72, attendance 68
pred = predict_one(new_student, w, b)

print("\nPrediction for new student: Exam Score =", new_student[0], "Attendance =", new_student[1])
if pred == 1:
    print("Result: Pass ✅ (The perceptron predicts this student WILL pass)")
else:
    print("Result: Fail ❌ (The perceptron predicts this student will NOT pass)")


# =========================================================
# 9) User Input Test Section
#    Allow user to enter new student data and display perceptron prediction.
# =========================================================

def test_with_user_input():
    print("\n=== Test the perceptron with your own input ===")
    try:
        user_exam = float(input("Enter exam score (e.g., 60): ").strip())
        user_attendance = float(input("Enter attendance percentage (e.g., 70): ").strip())
        user_sample = np.array([user_exam, user_attendance], dtype=float)
        user_pred = predict_one(user_sample, w, b)
        print(f"\nYour input: Exam Score = {user_exam}, Attendance = {user_attendance}")
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
    test_with_user_input()