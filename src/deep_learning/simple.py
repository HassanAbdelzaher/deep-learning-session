# 1) Install required packages (one time only):
# pip install scikit-learn joblib

# Import necessary functions and classes from scikit-learn and joblib
from sklearn.datasets import make_classification    # for creating a synthetic classification dataset
from sklearn.model_selection import train_test_split # for splitting the dataset into train and test sets
from sklearn.neural_network import MLPClassifier     # for constructing a multilayer perceptron neural network
from sklearn.metrics import accuracy_score, classification_report  # for evaluating model performance
import joblib                                        # for saving and loading the trained model

# -------------------------
# 2) Dataset creation (simple, synthetic example)
# -------------------------
# Generate a random classification dataset with 1000 samples, 10 features
#   n_informative = 6: number of useful (informative) features
#   n_redundant = 2: number of redundant (linearly dependent) features
#   n_classes = 2: binary classification (2 classes: 0 or 1)
#   random_state = 42: ensures results are reproducible (deterministic randomness)
X, y = make_classification(
    n_samples=1000,        # number of samples (rows)
    n_features=10,         # total number of features (columns)
    n_informative=6,       # number of informative features
    n_redundant=2,         # number of redundant features
    n_classes=2,           # target has 2 classes (binary)
    random_state=42        # random seed for reproducibility
)

# -------------------------
# 3) Split data into training and testing sets
# -------------------------
# Divide the dataset into 'train' and 'test' splits
#   test_size=0.2: 20% of the data for testing, 80% for training
#   random_state=42: reproducible splits
#   stratify=y: ensures class balance is preserved in splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# 4) Training a neural network model
# -------------------------
# Create a simple neural network (MLP) classifier with 2 hidden layers
#   hidden_layer_sizes=(16, 8): 16 neurons in first hidden, 8 in second hidden
#   activation="relu": rectified linear unit activation function
#   max_iter=500: train up to 500 epochs/iterations
#   random_state=42: reproducible initial weights
model = MLPClassifier(
    hidden_layer_sizes=(16, 8),  # architecture: 2 hidden layers
    activation="relu",           # activation for hidden layers
    max_iter=500,                # max training epochs
    random_state=42              # reproducibility
)

# Fit the neural network to the training data
model.fit(X_train, y_train)

# -------------------------
# 5) Testing / Evaluation
# -------------------------
# Use trained model to predict the labels for the test set
y_pred = model.predict(X_test)

# Calculate test accuracy (fraction of correct predictions)
acc = accuracy_score(y_test, y_pred)

# Print test accuracy, rounded to four decimal places
print("Test accuracy:", round(acc, 4))

# Print detailed classification metrics (precision, recall, f1-score, support)
print(classification_report(y_test, y_pred))

# -------------------------
# 6) Saving the trained model to disk
# -------------------------
# Save the trained model to a file named 'simple_nn_model.pkl' using joblib
joblib.dump(model, "simple_nn_model.pkl")
print("Saved model to simple_nn_model.pkl")

# -------------------------
# 7) Loading + Using the model for inference
# -------------------------
# Load the saved model back from disk
loaded_model = joblib.load("simple_nn_model.pkl")

# Example prediction: take 3 samples from X_test to make new predictions
new_samples = X_test[:3]                      # Select first three test samples
predictions = loaded_model.predict(new_samples)         # Get predicted class labels (0 or 1)
probabilities = loaded_model.predict_proba(new_samples) # Get probabilities for each class

# Print out predicted class labels for the new samples
print("Predictions:", predictions)
# Print probability estimates for each class for the new samples
print("Probabilities:", probabilities)
