"""
Very Simple Neural Network Example: Student Degree Classification

This script demonstrates a complete workflow:
1. Generate dataset
2. Train neural network
3. Test model
4. Save model
5. Load and use model for predictions
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import sys
from pathlib import Path
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import SimpleNeuralNetwork directly to avoid torch dependency
import importlib.util
spec = importlib.util.spec_from_file_location(
    "neural_networks",
    Path(__file__).parent.parent / "src" / "deep_learning" / "neural_networks.py"
)
neural_networks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(neural_networks)
SimpleNeuralNetwork = neural_networks.SimpleNeuralNetwork

print("=" * 70)
print("Simple Neural Network: Student Degree Classification")
print("=" * 70)

# ============================================================================
# 1. GENERATE DATASET
# ============================================================================
print("\n[1/5] Generating Dataset...")
np.random.seed(42)

n_students = 500
print(f"    Creating {n_students} student records...")

# Generate student features
attendance = np.random.uniform(60, 100, n_students)
quiz_avg = np.random.uniform(40, 95, n_students)
assignment_avg = np.random.uniform(45, 98, n_students)
midterm_score = np.random.uniform(35, 100, n_students)
project_score = np.random.uniform(50, 100, n_students)
study_hours = np.random.uniform(5, 40, n_students)

# Calculate final score
final_score = (
    0.10 * attendance +
    0.15 * quiz_avg +
    0.20 * assignment_avg +
    0.25 * midterm_score +
    0.20 * project_score +
    0.10 * (study_hours / 40 * 100)
)
final_score = np.clip(final_score + np.random.normal(0, 3, n_students), 0, 100)

# Classify into 5 degree categories
def get_degree_category(score):
    if score < 50:
        return 0  # Bad
    elif score < 65:
        return 1  # Acceptable
    elif score < 75:
        return 2  # Good
    elif score < 85:
        return 3  # Very Good
    else:
        return 4  # Excellent

y = np.array([get_degree_category(score) for score in final_score])
X = np.column_stack([attendance, quiz_avg, assignment_avg, midterm_score, 
                     project_score, study_hours])

degree_names = ['Bad', 'Acceptable', 'Good', 'Very Good', 'Excellent']
print(f"    Dataset created: {X.shape[0]} samples, {X.shape[1]} features")
print(f"    Classes: {len(degree_names)} categories")

# ============================================================================
# 2. PREPARE DATA (Split & Scale)
# ============================================================================
print("\n[2/5] Preparing Data...")

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert labels to one-hot encoding for neural network
def to_one_hot(y, num_classes=5):
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

y_train_onehot = to_one_hot(y_train)
y_test_onehot = to_one_hot(y_test)

print(f"    Training samples: {X_train_scaled.shape[0]}")
print(f"    Test samples: {X_test_scaled.shape[0]}")

# ============================================================================
# 3. TRAIN NEURAL NETWORK
# ============================================================================
print("\n[3/5] Training Neural Network...")
print("    Architecture: 6 inputs -> 16 hidden -> 8 hidden -> 5 outputs")

# Create neural network: 6 inputs, 2 hidden layers (16, 8), 5 outputs
nn = SimpleNeuralNetwork(layers=[6, 16, 8, 5], learning_rate=0.01)

# Train the model
print("    Training... (this may take a moment)")
loss_history = nn.train(
    X_train_scaled, 
    y_train_onehot, 
    epochs=1000, 
    verbose=True
)

print(f"    Training complete! Final loss: {loss_history[-1]:.6f}")

# ============================================================================
# 4. TEST MODEL
# ============================================================================
print("\n[4/5] Testing Model...")

# Make predictions
y_pred_proba = nn.predict(X_test_scaled)
y_pred = np.argmax(y_pred_proba, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"    Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Classification report
print("\n    Classification Report:")
print(classification_report(y_test, y_pred, target_names=degree_names, 
                          labels=[0, 1, 2, 3, 4], zero_division=0))

# ============================================================================
# 5. SAVE MODEL
# ============================================================================
print("\n[5/5] Saving Model...")

model_dir = Path('models')
model_dir.mkdir(exist_ok=True)

# Save model weights, biases, and architecture
model_path = model_dir / 'simple_student_nn.pkl'
scaler_path = model_dir / 'simple_student_scaler.pkl'

# Save model state (weights, biases, layers, learning_rate)
model_state = {
    'weights': nn.weights,
    'biases': nn.biases,
    'layers': nn.layers,
    'learning_rate': nn.learning_rate
}

with open(model_path, 'wb') as f:
    pickle.dump(model_state, f)
print(f"    Model saved to: {model_path}")

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"    Scaler saved to: {scaler_path}")

# ============================================================================
# 6. LOAD AND USE MODEL (INFERENCE)
# ============================================================================
print("\n" + "=" * 70)
print("Using Saved Model for Predictions")
print("=" * 70)

# Load model and scaler
print("\n[Loading] Loading saved model...")
with open(model_path, 'rb') as f:
    model_state = pickle.load(f)

with open(scaler_path, 'rb') as f:
    loaded_scaler = pickle.load(f)

# Recreate neural network from saved state
loaded_nn = SimpleNeuralNetwork(
    layers=model_state['layers'],
    learning_rate=model_state['learning_rate']
)
loaded_nn.weights = model_state['weights']
loaded_nn.biases = model_state['biases']

print("    Model loaded successfully!")

# Example: Predict on new student data
print("\n[Prediction] Predicting on new student data...")

# Example students
new_students = np.array([
    [95, 90, 88, 92, 85, 30],  # High scores -> Excellent
    [70, 60, 65, 58, 70, 15],  # Medium scores -> Acceptable
    [50, 45, 50, 48, 55, 10],  # Low scores -> Bad
])

# Scale the new data
new_students_scaled = loaded_scaler.transform(new_students)

# Make predictions
predictions_proba = loaded_nn.predict(new_students_scaled)
predictions = np.argmax(predictions_proba, axis=1)

print("\n    Predictions:")
print("    " + "-" * 60)
for i, (student, pred, proba) in enumerate(zip(new_students, predictions, predictions_proba), 1):
    print(f"\n    Student {i}:")
    print(f"      Features: Attendance={student[0]:.1f}%, Quiz={student[1]:.1f}, "
          f"Assignment={student[2]:.1f}, Midterm={student[3]:.1f}, "
          f"Project={student[4]:.1f}, Study Hours={student[5]:.1f}")
    print(f"      Predicted Degree: {degree_names[pred]}")
    print(f"      Confidence: {proba[pred]*100:.1f}%")
    print(f"      Probabilities: {dict(zip(degree_names, [f'{p*100:.1f}%' for p in proba]))}")

print("\n" + "=" * 70)
print("Complete! Model trained, tested, saved, and used successfully.")
print("=" * 70)
