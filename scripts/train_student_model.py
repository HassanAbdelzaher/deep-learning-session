"""
Train and test the student degree classification model
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly from module to avoid torch dependency
import importlib.util
spec = importlib.util.spec_from_file_location(
    "neural_networks",
    Path(__file__).parent.parent / "src" / "deep_learning" / "neural_networks.py"
)
neural_networks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(neural_networks)
SimpleNeuralNetwork = neural_networks.SimpleNeuralNetwork

print("=" * 60)
print("Student Degree Classification Model Training")
print("=" * 60)

# Load dataset
print("\n[1/5] Loading dataset...")
df = pd.read_csv('data/neural_networks/student_degree_dataset.csv')
print(f"    Dataset loaded: {df.shape[0]} students")

# Prepare features and labels
print("\n[2/5] Preparing data...")
feature_columns = ['attendance', 'quiz_avg', 'assignment_avg', 'midterm_score',
                   'project_score', 'study_hours_per_week', 'participation_score']
X = df[feature_columns].values
y_categories = df['degree_category'].values

# Map categories to numbers
category_mapping = {
    'Bad': 0,
    'Acceptable': 1,
    'Good': 2,
    'Very Good': 3,
    'Excellent': 4
}
y = np.array([category_mapping[cat] for cat in y_categories])

# One-hot encode
def one_hot_encode(y, num_classes=5):
    encoded = np.zeros((len(y), num_classes))
    encoded[np.arange(len(y)), y] = 1
    return encoded

y_encoded = one_hot_encode(y)

# Split data (use stratify only if all classes have at least 2 samples)
unique, counts = np.unique(y, return_counts=True)
min_class_count = counts.min()
if min_class_count >= 2:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y
    )
else:
    print(f"    Warning: Some classes have < 2 samples. Using non-stratified split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"    Training set: {X_train.shape[0]} samples")
print(f"    Test set: {X_test.shape[0]} samples")
print(f"    Features: {X_train.shape[1]}")
print(f"    Classes: {y_encoded.shape[1]}")

# Create and train neural network
print("\n[3/5] Creating neural network...")
nn = SimpleNeuralNetwork(layers=[7, 32, 16, 5], learning_rate=0.01)
print("    Architecture: 7 -> 32 -> 16 -> 5")

print("\n[4/5] Training model...")
loss_history = nn.train(X_train, y_train, epochs=200, verbose=False)
print(f"    Training complete! Final loss: {loss_history[-1]:.6f}")

# Evaluate
print("\n[5/5] Evaluating model...")
predictions = nn.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = np.argmax(y_test, axis=1)
accuracy = accuracy_score(actual_classes, predicted_classes)

print(f"\n" + "=" * 60)
print("Model Evaluation Results")
print("=" * 60)
print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Classification report
category_names = ['Bad', 'Acceptable', 'Good', 'Very Good', 'Excellent']
print("\nClassification Report:")
print(classification_report(actual_classes, predicted_classes, 
                          target_names=category_names, 
                          labels=[0, 1, 2, 3, 4],
                          zero_division=0))

# Confusion matrix
cm = confusion_matrix(actual_classes, predicted_classes)
print("Confusion Matrix:")
print(cm)

# Save model info
model_info = {
    'accuracy': float(accuracy),
    'final_loss': float(loss_history[-1]),
    'architecture': [7, 32, 16, 5],
    'training_samples': int(X_train.shape[0]),
    'test_samples': int(X_test.shape[0])
}

import json
with open('data/neural_networks/student_model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print(f"\n[OK] Model info saved to: data/neural_networks/student_model_info.json")
print("=" * 60)
print("Training and evaluation complete!")
