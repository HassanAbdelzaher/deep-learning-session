"""
Test TensorFlow/Keras student degree classification model
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import sys
from pathlib import Path
import json

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import to_categorical
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("ERROR: TensorFlow not installed. Install with: pip install tensorflow")
    sys.exit(1)

print("=" * 70)
print("TensorFlow Student Degree Classification Model Testing")
print("=" * 70)

# Load dataset
print("\n[1/3] Loading dataset...")
df = pd.read_csv('data/neural_networks/student_degree_dataset.csv')
feature_columns = ['attendance', 'quiz_avg', 'assignment_avg', 'midterm_score',
                   'project_score', 'study_hours_per_week', 'participation_score']
X = df[feature_columns].values

# Create polynomial features (same as training)
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
poly.fit(X)

# Create scaler (same as training)
scaler = StandardScaler()
scaler.fit(poly.transform(X))

# Load model info
try:
    with open('data/neural_networks/tensorflow_model_info.json', 'r') as f:
        model_info = json.load(f)
    print(f"    Model architecture: {model_info['architecture']}")
    print(f"    Training accuracy: {model_info['accuracy']:.4f}")
except:
    print("    [WARNING] Model info not found. Model may not be trained yet.")
    model_info = None

# Load model
print("\n[2/3] Loading model...")
try:
    model = load_model('data/neural_networks/best_tensorflow_model.h5')
    print("    Model loaded successfully!")
except:
    print("    [ERROR] Model file not found. Please train the model first with:")
    print("           make train-tensorflow-student-model")
    sys.exit(1)

# Test on sample students
print("\n[3/3] Testing on sample students...")
category_names = ['Bad', 'Acceptable', 'Good', 'Very Good', 'Excellent']

test_students = [
    {'name': 'Excellent Student', 'features': [95.0, 90.0, 92.0, 88.0, 95.0, 35.0, 95.0]},
    {'name': 'Very Good Student', 'features': [88.0, 82.0, 85.0, 80.0, 88.0, 28.0, 85.0]},
    {'name': 'Good Student', 'features': [80.0, 75.0, 78.0, 72.0, 80.0, 20.0, 75.0]},
    {'name': 'Acceptable Student', 'features': [70.0, 60.0, 65.0, 58.0, 62.0, 15.0, 60.0]},
    {'name': 'Struggling Student', 'features': [55.0, 45.0, 50.0, 48.0, 52.0, 10.0, 50.0]}
]

print("\n" + "=" * 70)
print("Test Results")
print("=" * 70)

for student in test_students:
    features = np.array([student['features']])
    features_poly = poly.transform(features)
    features_scaled = scaler.transform(features_poly)
    
    prediction = model.predict(features_scaled, verbose=0)
    predicted_class_idx = np.argmax(prediction)
    predicted_category = category_names[predicted_class_idx]
    confidence = prediction[0][predicted_class_idx]
    
    print(f"\n{student['name']}:")
    print(f"  Features: {student['features']}")
    print(f"  Predicted: {predicted_category} (confidence: {confidence:.4f} = {confidence*100:.2f}%)")
    print(f"  Probabilities:")
    for i, cat in enumerate(category_names):
        print(f"    {cat:15s}: {prediction[0][i]:.4f} ({prediction[0][i]*100:5.2f}%)")

print("\n" + "=" * 70)
print("[OK] Model testing complete!")
print("=" * 70)
