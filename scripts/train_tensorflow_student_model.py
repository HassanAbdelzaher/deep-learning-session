"""
Train TensorFlow/Keras student degree classification model
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import sys
from pathlib import Path
import json

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks, optimizers
    from tensorflow.keras.utils import to_categorical
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("ERROR: TensorFlow not installed. Install with: pip install tensorflow")
    sys.exit(1)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("TensorFlow Student Degree Classification Model Training")
print("=" * 70)

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Load dataset
print("\n[1/6] Loading dataset...")
df = pd.read_csv('data/neural_networks/student_degree_dataset.csv')
print(f"    Dataset loaded: {df.shape[0]} students")

# Prepare features and labels
print("\n[2/6] Preparing data...")
feature_columns = ['attendance', 'quiz_avg', 'assignment_avg', 'midterm_score',
                   'project_score', 'study_hours_per_week', 'participation_score']
X = df[feature_columns].values
y_categories = df['degree_category'].values

category_mapping = {'Bad': 0, 'Acceptable': 1, 'Good': 2, 'Very Good': 3, 'Excellent': 4}
y = np.array([category_mapping[cat] for cat in y_categories])

# Feature Engineering: Polynomial features
print("    Creating polynomial features...")
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_poly = poly.fit_transform(X)
print(f"    Features: {X.shape[1]} → {X_poly.shape[1]} (polynomial)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One-hot encode
y_train_cat = to_categorical(y_train, 5)
y_test_cat = to_categorical(y_test, 5)

# Class balancing with oversampling
print("\n[3/6] Balancing classes...")
def oversample_minority_classes(X, y, y_cat, target_samples=300):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    X_balanced = [X]
    y_balanced = [y]
    y_cat_balanced = [y_cat]
    
    for class_idx, count in zip(unique_classes, class_counts):
        if count < target_samples:
            class_mask = y == class_idx
            X_class = X[class_mask]
            y_class = y[class_mask]
            y_cat_class = y_cat[class_mask]
            n_samples_needed = target_samples - count
            indices = np.random.choice(len(X_class), size=n_samples_needed, replace=True)
            X_balanced.append(X_class[indices])
            y_balanced.append(y_class[indices])
            y_cat_balanced.append(y_cat_class[indices])
    
    X_balanced = np.vstack(X_balanced)
    y_balanced = np.hstack(y_balanced)
    y_cat_balanced = np.vstack(y_cat_balanced)
    shuffle_idx = np.random.permutation(len(X_balanced))
    return X_balanced[shuffle_idx], y_balanced[shuffle_idx], y_cat_balanced[shuffle_idx]

X_train_balanced, y_train_balanced, y_train_cat_balanced = oversample_minority_classes(
    X_train_scaled, y_train, y_train_cat, target_samples=300
)

# Create validation set
X_train_final, X_val, y_train_final, y_val, y_train_final_cat, y_val_cat = train_test_split(
    X_train_balanced, y_train_balanced, y_train_cat_balanced,
    test_size=0.15, random_state=42, stratify=y_train_balanced
)

print(f"    Training: {X_train_final.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test_scaled.shape[0]}")

# Build model
print("\n[4/6] Building TensorFlow model...")
input_size = X_train_final.shape[1]
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(input_size,)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(5, activation='softmax')
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"    Architecture: {input_size} → 128 → 256 → 128 → 64 → 5")

# Setup callbacks
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001, verbose=1),
    callbacks.ModelCheckpoint('data/neural_networks/best_tensorflow_model.h5', 
                             monitor='val_accuracy', save_best_only=True, verbose=1)
]

# Train model
print("\n[5/6] Training model...")
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_final), y=y_train_final)
class_weight_dict = dict(enumerate(class_weights))

history = model.fit(
    X_train_final, y_train_final_cat,
    validation_data=(X_val, y_val_cat),
    epochs=300,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=callbacks_list,
    verbose=1
)

# Load best model
model.load_weights('data/neural_networks/best_tensorflow_model.h5')

# Evaluate
print("\n[6/6] Evaluating model...")
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
predictions = model.predict(X_test_scaled, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = np.argmax(y_test_cat, axis=1)

print(f"\n" + "=" * 70)
print("Model Evaluation Results")
print("=" * 70)
print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Test Loss: {test_loss:.6f}")

# Save model info
model_info = {
    'framework': 'TensorFlow/Keras',
    'accuracy': float(test_accuracy),
    'test_loss': float(test_loss),
    'architecture': [int(input_size), 128, 256, 128, 64, 5],
    'training_samples': int(X_train_final.shape[0]),
    'validation_samples': int(X_val.shape[0]),
    'test_samples': int(X_test_scaled.shape[0]),
    'best_val_accuracy': float(max(history.history['val_accuracy']))
}

with open('data/neural_networks/tensorflow_model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print(f"\n[OK] Model saved to: data/neural_networks/best_tensorflow_model.h5")
print(f"[OK] Model info saved to: data/neural_networks/tensorflow_model_info.json")
print("=" * 70)
print("Training complete!")
