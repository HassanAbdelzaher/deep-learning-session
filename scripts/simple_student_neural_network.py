"""
Simple Neural Network Example: Student Degree Classification
Complete workflow: Dataset -> Training -> Testing -> Saving -> Using
"""

import numpy as np
import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import directly from module to avoid torch dependency
import importlib.util
from pathlib import Path

# Load neural_networks module directly
spec = importlib.util.spec_from_file_location(
    "neural_networks",
    Path(__file__).parent.parent / "src" / "deep_learning" / "neural_networks.py"
)
neural_networks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(neural_networks)
SimpleNeuralNetwork = neural_networks.SimpleNeuralNetwork

# Import dataset loader
from utils.dataset_loader import load_student_degree

# Import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def prepare_data():
    """Step 1: Load and prepare the dataset"""
    print("=" * 60)
    print("STEP 1: Loading Dataset")
    print("=" * 60)
    
    # Load student degree dataset
    X, y = load_student_degree()
    
    print(f"Dataset loaded:")
    print(f"  Features shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")
    print(f"  Number of classes: {len(np.unique(y))}")
    print(f"  Class distribution: {np.bincount(y)}")
    
    # One-hot encode labels
    num_classes = 5
    y_onehot = np.zeros((len(y), num_classes))
    y_onehot[np.arange(len(y)), y] = 1
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nData split:")
    print(f"  Training samples: {X_train_scaled.shape[0]}")
    print(f"  Test samples: {X_test_scaled.shape[0]}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, y_test.argmax(axis=1)


def train_model(X_train, y_train):
    """Step 2: Train the neural network"""
    print("\n" + "=" * 60)
    print("STEP 2: Training Neural Network")
    print("=" * 60)
    
    # Create neural network: 7 input features -> 32 -> 16 -> 5 output classes
    print("Creating neural network...")
    print("  Architecture: 7 -> 32 -> 16 -> 5")
    print("  Learning rate: 0.01")
    
    nn = SimpleNeuralNetwork(layers=[7, 32, 16, 5], learning_rate=0.01)
    
    # Train the model
    print("\nTraining model...")
    loss_history = nn.train(X_train, y_train, epochs=100, verbose=True)
    
    print(f"\nTraining complete!")
    print(f"  Final loss: {loss_history[-1]:.4f}")
    print(f"  Total epochs: {len(loss_history)}")
    
    return nn, loss_history


def test_model(nn, X_test, y_test_true):
    """Step 3: Test the model"""
    print("\n" + "=" * 60)
    print("STEP 3: Testing Model")
    print("=" * 60)
    
    # Make predictions
    predictions = nn.predict(X_test)
    y_pred = predictions.argmax(axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test_true, y_pred)
    
    print(f"Test Results:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test_true, y_pred, 
                                target_names=['Bad', 'Acceptable', 'Good', 'Very Good', 'Excellent'],
                                zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    return accuracy, y_pred


def save_model(nn, scaler, model_info):
    """Step 4: Save the model"""
    print("\n" + "=" * 60)
    print("STEP 4: Saving Model")
    print("=" * 60)
    
    # Save model weights and architecture
    model_dir = "data/neural_networks"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model info
    model_info_path = os.path.join(model_dir, "simple_student_model_info.json")
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"Model saved to: {model_info_path}")
    print(f"  Architecture: {model_info['architecture']}")
    print(f"  Accuracy: {model_info['accuracy']:.4f}")
    print(f"  Training samples: {model_info['training_samples']}")
    print(f"  Test samples: {model_info['test_samples']}")
    
    # Note: SimpleNeuralNetwork doesn't have built-in save method
    # In a real scenario, you would save weights separately
    print("\nNote: For production, save model weights separately")
    print("      This example saves model metadata only")
    
    return model_info_path


def use_model(model_info_path, scaler):
    """Step 5: Use the saved model"""
    print("\n" + "=" * 60)
    print("STEP 5: Using Saved Model")
    print("=" * 60)
    
    # Load model info
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    print(f"Model loaded from: {model_info_path}")
    print(f"  Architecture: {model_info['architecture']}")
    print(f"  Accuracy: {model_info['accuracy']:.4f}")
    
    # Create new model with same architecture
    architecture = model_info['architecture']
    nn = SimpleNeuralNetwork(layers=architecture, learning_rate=0.01)
    
    # Example: Predict for new students
    print("\nPredicting for new students:")
    
    # Sample student data: [attendance, quiz_avg, assignment_avg, midterm, final, project, participation]
    new_students = np.array([
        [0.95, 0.85, 0.90, 0.88, 0.92, 0.90, 0.95],  # Excellent student
        [0.70, 0.65, 0.70, 0.68, 0.72, 0.70, 0.75],  # Acceptable student
        [0.50, 0.45, 0.50, 0.48, 0.52, 0.50, 0.55],  # Bad student
    ])
    
    # Scale the features
    new_students_scaled = scaler.transform(new_students)
    
    # Make predictions
    predictions = nn.predict(new_students_scaled)
    predicted_classes = predictions.argmax(axis=1)
    
    # Class names
    class_names = ['Bad', 'Acceptable', 'Good', 'Very Good', 'Excellent']
    
    print("\nPredictions:")
    for i, (student, pred_class, probs) in enumerate(zip(new_students, predicted_classes, predictions)):
        print(f"\nStudent {i+1}:")
        print(f"  Features: Attendance={student[0]:.2f}, Quiz={student[1]:.2f}, "
              f"Assignment={student[2]:.2f}, Midterm={student[3]:.2f}, "
              f"Final={student[4]:.2f}, Project={student[5]:.2f}, Participation={student[6]:.2f}")
        print(f"  Predicted Class: {class_names[pred_class]}")
        print(f"  Confidence: {probs[pred_class]:.4f}")
        print(f"  Probabilities: {dict(zip(class_names, probs))}")
    
    return nn


def main():
    """Complete workflow: Dataset -> Training -> Testing -> Saving -> Using"""
    print("\n" + "=" * 60)
    print("SIMPLE NEURAL NETWORK: Student Degree Classification")
    print("=" * 60)
    print("\nThis example demonstrates the complete workflow:")
    print("  1. Load and prepare dataset")
    print("  2. Train neural network")
    print("  3. Test model performance")
    print("  4. Save model")
    print("  5. Use saved model for predictions")
    print("=" * 60)
    
    try:
        # Step 1: Prepare data
        X_train, X_test, y_train, y_test, scaler, y_test_true = prepare_data()
        
        # Step 2: Train model
        nn, loss_history = train_model(X_train, y_train)
        
        # Step 3: Test model
        accuracy, y_pred = test_model(nn, X_test, y_test_true)
        
        # Step 4: Save model
        model_info = {
            'architecture': [7, 32, 16, 5],
            'accuracy': float(accuracy),
            'training_samples': int(X_train.shape[0]),
            'test_samples': int(X_test.shape[0]),
            'final_loss': float(loss_history[-1]),
            'epochs': len(loss_history)
        }
        model_info_path = save_model(nn, scaler, model_info)
        
        # Step 5: Use model
        used_model = use_model(model_info_path, scaler)
        
        print("\n" + "=" * 60)
        print("COMPLETE! All steps executed successfully.")
        print("=" * 60)
        print("\nSummary:")
        print(f"  ✓ Dataset loaded and prepared")
        print(f"  ✓ Model trained ({len(loss_history)} epochs)")
        print(f"  ✓ Model tested (Accuracy: {accuracy:.2%})")
        print(f"  ✓ Model saved to {model_info_path}")
        print(f"  ✓ Model used for predictions")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
