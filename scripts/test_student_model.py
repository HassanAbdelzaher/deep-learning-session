"""
Test the student degree classification model on new data
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import json
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
print("Student Degree Classification Model Testing")
print("=" * 60)

# Load dataset to get scaler
print("\n[1/4] Loading dataset and preparing scaler...")
df = pd.read_csv('data/neural_networks/student_degree_dataset.csv')
feature_columns = ['attendance', 'quiz_avg', 'assignment_avg', 'midterm_score',
                   'project_score', 'study_hours_per_week', 'participation_score']
X = df[feature_columns].values

# Create scaler (same as training)
scaler = StandardScaler()
scaler.fit(X)

# Load model info
try:
    with open('data/neural_networks/student_model_info.json', 'r') as f:
        model_info = json.load(f)
    print(f"    Model architecture: {model_info['architecture']}")
    print(f"    Training accuracy: {model_info['accuracy']:.4f}")
except:
    print("    [WARNING] Model info not found. Training new model...")
    model_info = {'architecture': [7, 32, 16, 5]}

# Create model (weights will be random if not trained)
print("\n[2/4] Creating model...")
nn = SimpleNeuralNetwork(layers=model_info['architecture'], learning_rate=0.01)

# Train model quickly for testing
print("\n[3/4] Training model (quick training for testing)...")
from sklearn.model_selection import train_test_split
y_categories = df['degree_category'].values
category_mapping = {'Bad': 0, 'Acceptable': 1, 'Good': 2, 'Very Good': 3, 'Excellent': 4}
y = np.array([category_mapping[cat] for cat in y_categories])

def one_hot_encode(y, num_classes=5):
    encoded = np.zeros((len(y), num_classes))
    encoded[np.arange(len(y)), y] = 1
    return encoded

y_encoded = one_hot_encode(y)
# Split data (use stratify only if all classes have at least 2 samples)
unique, counts = np.unique(y, return_counts=True)
min_class_count = counts.min()
if min_class_count >= 2:
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn.train(X_train_scaled, y_train, epochs=50, verbose=False)

# Test on sample students
print("\n[4/4] Testing model on sample students...")
category_names = ['Bad', 'Acceptable', 'Good', 'Very Good', 'Excellent']

# Test cases
test_students = [
    {
        'name': 'Excellent Student',
        'features': [95.0, 90.0, 92.0, 88.0, 95.0, 35.0, 95.0]
    },
    {
        'name': 'Good Student',
        'features': [80.0, 75.0, 78.0, 72.0, 80.0, 20.0, 75.0]
    },
    {
        'name': 'Acceptable Student',
        'features': [70.0, 60.0, 65.0, 58.0, 62.0, 15.0, 60.0]
    },
    {
        'name': 'Struggling Student',
        'features': [55.0, 45.0, 50.0, 48.0, 52.0, 10.0, 50.0]
    }
]

print("\n" + "=" * 60)
print("Test Results")
print("=" * 60)

for student in test_students:
    features = np.array([student['features']])
    features_scaled = scaler.transform(features)
    prediction = nn.predict(features_scaled)
    predicted_class_idx = np.argmax(prediction)
    predicted_category = category_names[predicted_class_idx]
    confidence = prediction[0][predicted_class_idx]
    
    print(f"\n{student['name']}:")
    print(f"  Features: {student['features']}")
    print(f"  Predicted: {predicted_category} (confidence: {confidence:.4f})")
    print(f"  Probabilities:")
    for i, cat in enumerate(category_names):
        print(f"    {cat:15s}: {prediction[0][i]:.4f}")

# Test on actual test set
predictions = nn.predict(X_test_scaled)
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = np.argmax(y_test, axis=1)
test_accuracy = np.mean(predicted_classes == actual_classes)

print(f"\n" + "=" * 60)
print(f"Test Set Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print("=" * 60)
print("\n[OK] Model testing complete!")
