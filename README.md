# Python AI Learning Sessions

A comprehensive Python project for learning mathematics and deep learning concepts through hands-on sessions, complete with visualizations, applied projects, and automated workflows. Features both custom neural network implementations and production-ready TensorFlow/Keras models achieving >90% accuracy.

## Project Structure

```
py-ai/
├── src/                        # Source code modules
│   ├── mathematics/            # Mathematical concepts and utilities
│   │   ├── __init__.py
│   │   ├── linear_algebra.py   # Vector/matrix operations, visualizations
│   │   ├── calculus.py         # Derivatives, gradients, optimization
│   │   └── statistics.py       # Distributions, hypothesis testing
│   ├── deep_learning/          # Deep learning implementations
│   │   ├── __init__.py
│   │   ├── neural_networks.py  # MLP, backpropagation, training
│   │   ├── cnn.py              # Convolutional neural networks
│   │   └── rnn.py              # Recurrent networks, LSTMs
│   └── utils/                  # Helper functions and utilities
│       ├── __init__.py
│       ├── data_loader.py      # Dataset loading utilities
│       ├── dataset_loader.py   # Alternative dataset loader
│       └── visualization.py    # Visualization helpers
│
├── docs/                       # Comprehensive documentation with graphs
│   ├── README.md               # Documentation index
│   ├── 01_linear_algebra.md    # Linear algebra concepts
│   ├── 02_calculus.md          # Calculus and optimization
│   ├── 03_statistics.md        # Statistics and probability
│   ├── 04_neural_networks.md   # Neural network fundamentals
│   ├── 05_cnns.md              # Convolutional neural networks
│   ├── 06_rnns.md              # Recurrent neural networks
│   ├── projects/               # Applied project documentation
│   │   ├── README.md
│   │   ├── 01_applied_linear_algebra_projects.md
│   │   ├── 02_applied_neural_network_projects.md
│   │   └── PROJECT_7_SUMMARY.md
│   └── images/                 # Generated visualization images
│
├── notebooks/                  # Jupyter notebooks for interactive learning
│   ├── 01_mathematics_basics.ipynb
│   ├── 02_neural_networks.ipynb
│   ├── 03_project_1_mnist_digit_recognition.ipynb
│   ├── 04_project_2_spam_detection.ipynb
│   ├── 05_project_3_iris_classification.ipynb
│   ├── 06_project_4_house_price_prediction.ipynb
│   ├── 07_project_5_xor_problem.ipynb
│   ├── 08_project_6_cnn_image_classification.ipynb
│   ├── 09_project_7_student_degree_classification.ipynb
│   ├── 10_project_8_student_degree_classification_advanced.ipynb
│   └── 11_project_9_student_degree_classification_tensorflow.ipynb
│
├── scripts/                    # Automation and utility scripts
│   ├── generate_neural_network_datasets.py  # Generate all NN datasets
│   ├── generate_student_dataset.py          # Generate student dataset
│   ├── train_student_model.py               # Train custom NN student classifier
│   ├── test_student_model.py                # Test custom NN student classifier
│   ├── train_tensorflow_student_model.py    # Train TensorFlow/Keras student classifier
│   ├── test_tensorflow_student_model.py     # Test TensorFlow/Keras student classifier
│   └── test_dataset_loader.py               # Test dataset loading
│
├── data/                       # Datasets and data files
│   ├── README.md
│   └── neural_networks/        # Generated datasets for NN projects
│       ├── README.md
│       ├── mnist_X.npy         # MNIST features (excluded from git)
│       ├── mnist_y.npy         # MNIST labels (excluded from git)
│       ├── iris_X.npy          # Iris dataset features
│       ├── iris_y.npy           # Iris dataset labels
│       ├── iris_metadata.pkl   # Iris metadata
│       ├── spam_X.npy          # Spam detection features
│       ├── spam_y.npy           # Spam detection labels
│       ├── house_X.npy         # House price features
│       ├── house_y.npy         # House price labels
│       ├── house_metadata.pkl  # House price metadata
│       ├── xor_X.npy           # XOR problem features
│       ├── xor_y.npy           # XOR problem labels
│       ├── cnn_X.npy           # CNN image features
│       ├── cnn_y.npy           # CNN image labels
│       ├── student_degree_dataset.csv  # Student classification CSV
│       ├── student_X.npy        # Student features
│       ├── student_y_scores.npy # Student scores
│       ├── student_y_categories.npy # Student categories
│       ├── student_y_encoded.npy # One-hot encoded labels
│       ├── student_model_info.json # Custom NN model metadata (Project 7)
│       ├── best_tensorflow_model.h5 # TensorFlow/Keras trained model (Project 9)
│       ├── tensorflow_model_info.json # TensorFlow model metadata (Project 9)
│       └── training_history.csv # TensorFlow training history logs
│
├── tests/                      # Unit tests
│   ├── __init__.py
│   └── test_mathematics.py
│
├── sessions/                   # Learning session modules
│   └── README.md
│
├── Makefile                    # Automation commands (see make help)
├── MAKEFILE_USAGE.md          # Makefile usage documentation
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup configuration
├── generate_docs_graphs.py     # Script to generate documentation graphs
├── quick_start.py             # Quick start script
└── .gitignore                  # Git ignore rules
```

## Quick Start

### Option 1: Using Makefile (Recommended)

The project includes a comprehensive Makefile for easy automation:

```bash
# Complete setup (install dependencies, generate datasets, run tests)
make setup

# Or step by step:
make install          # Install dependencies
make datasets         # Generate all datasets
make student-dataset  # Generate student classification dataset
make notebooks        # Launch Jupyter Lab
```

**See all available commands:**
```bash
make help
```

### Option 2: Manual Setup

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment:**
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate datasets:**
   ```bash
   python scripts/generate_neural_network_datasets.py
   python scripts/generate_student_dataset.py
   ```

5. **Start Jupyter Lab:**
   ```bash
   jupyter lab
   ```

## Documentation

**Comprehensive learning guides with visualizations!** See [docs/README.md](docs/README.md) for the full index.

### 🗺️ [Learning Roadmap](docs/LEARNING_ROADMAP.md)

**New to the project?** Start with the [Learning Roadmap](docs/LEARNING_ROADMAP.md) - a comprehensive guide with structured learning paths from beginner to advanced, timeline estimates, and project recommendations.

### Mathematics Documentation
- **[Linear Algebra](docs/01_linear_algebra.md)** - Vectors, matrices, eigenvalues with graphs
- **[Calculus](docs/02_calculus.md)** - Derivatives, gradients, optimization with visualizations
- **[Statistics](docs/03_statistics.md)** - Distributions, hypothesis testing, normalization

### Deep Learning Documentation
- **[Neural Networks](docs/04_neural_networks.md)** - Architecture, forward/backward propagation
- **[CNNs](docs/05_cnns.md)** - Convolution, pooling, feature learning
- **[RNNs](docs/06_rnns.md)** - Recurrent networks, LSTMs, sequence processing

### Applied Projects

#### Linear Algebra Projects
- **[Applied Linear Algebra Projects](docs/projects/01_applied_linear_algebra_projects.md)** - 6 hands-on projects:
  - Image Transformations
  - Principal Component Analysis (PCA)
  - Linear Regression from Scratch
  - Simple Neural Network
  - Data Preprocessing Pipeline
  - Face Recognition with Eigenfaces

#### Neural Network Projects
- **[Applied Neural Network Projects](docs/projects/02_applied_neural_network_projects.md)** - 9 practical projects:
  - **Project 1: MNIST Digit Recognition** - Handwritten digit classification
  - **Project 2: Spam Detection** - Binary text classification
  - **Project 3: Iris Classification** - Multi-class flower classification
  - **Project 4: House Price Prediction** - Regression problem
  - **Project 5: XOR Problem** - Non-linearity demonstration
  - **Project 6: CNN Image Classification** - Convolutional neural networks
- **Project 7: Student Degree Classification** - Real-world CSV data classification (Custom NN, ~62% accuracy)
- **Project 8: Student Degree Classification (Advanced)** - High-accuracy with feature engineering (Custom NN, ~85% accuracy)
- **Project 9: Student Degree Classification (TensorFlow)** - Production-ready TensorFlow/Keras implementation (>90% accuracy)
  - **Project 9: Student Degree Classification (TensorFlow)** - Production-ready TensorFlow/Keras

Each documentation file includes:
- Detailed explanations
- Code examples with visualizations
- Graphs and diagrams
- Practice exercises
- Step-by-step tutorials

## Jupyter Notebooks

Interactive notebooks for hands-on learning:

### Basics
- **01_mathematics_basics.ipynb** - Linear algebra, calculus, and statistics fundamentals
- **02_neural_networks.ipynb** - Neural network architecture and training

### Applied Projects
- **03_project_1_mnist_digit_recognition.ipynb** - MNIST digit classification
- **04_project_2_spam_detection.ipynb** - Binary spam detection
- **05_project_3_iris_classification.ipynb** - Multi-class Iris classification
- **06_project_4_house_price_prediction.ipynb** - Regression with house prices
- **07_project_5_xor_problem.ipynb** - Solving XOR with neural networks
- **08_project_6_cnn_image_classification.ipynb** - CNN for image classification
- **09_project_7_student_degree_classification.ipynb** - Student degree classification from CSV (Custom NN, ~62% accuracy)
- **10_project_8_student_degree_classification_advanced.ipynb** - Advanced high-accuracy classification (Custom NN, ~85% accuracy)
- **11_project_9_student_degree_classification_tensorflow.ipynb** - TensorFlow/Keras implementation (>90% accuracy)

## Learning Sessions

### Mathematics Sessions
- Linear Algebra fundamentals (vectors, matrices, transformations)
- Calculus and optimization (derivatives, gradients, gradient descent)
- Probability and statistics (distributions, hypothesis testing)
- Numerical methods

### Deep Learning Sessions
- Neural Networks basics (perceptron, MLP, backpropagation)
- Convolutional Neural Networks (CNNs) - Image processing
- Recurrent Neural Networks (RNNs) - Sequence processing
- Applied projects with real datasets

## Usage

### Using Makefile Commands

The project includes automated workflows via Makefile:

```bash
# Dataset Management
make datasets              # Generate all neural network datasets
make student-dataset       # Generate student classification dataset
make test-datasets         # Test dataset loader functionality

# Student Classification Models
make train-student-model          # Train custom NN student model
make test-student-model           # Test custom NN student model
make student-model                # Complete custom NN pipeline
make train-tensorflow-student-model  # Train TensorFlow/Keras model (>90% accuracy)
make test-tensorflow-student-model   # Test TensorFlow/Keras model
make tensorflow-student-model        # Complete TensorFlow pipeline

# Development
make install              # Install dependencies
make install-dev          # Install with development tools
make lint                 # Run code linters
make format               # Format code with black
make test                 # Run all tests
make clean                # Clean generated files
make clean-models         # Clean trained model files (.h5, .json)
make clean-all            # Clean everything (files, datasets, models)

# Notebooks
make notebooks            # Launch Jupyter Lab

# Information
make info                 # Show project information
make help                 # Show all available commands
```

### Using the Code Modules

Each session module can be run independently or imported as a library:

```python
from src.mathematics import linear_algebra
from src.deep_learning import neural_networks
from src.utils.data_loader import load_mnist, load_iris

# Run examples
linear_algebra.visualize_vector_addition()
neural_networks.xor_problem_example()

# Load datasets
X, y = load_mnist()
X, y, metadata = load_iris()
```

### Using the Documentation

1. **Read the markdown files** in `docs/` for comprehensive explanations
2. **Run code examples** from the documentation to generate visualizations
3. **Study the graphs** to understand concepts visually
4. **Complete exercises** at the end of each section
5. **Follow along with Jupyter notebooks** for interactive learning

Example: Open `docs/01_linear_algebra.md` and run the Python code blocks to generate graphs showing vector operations, matrix multiplications, and more!

### Student Classification Models

Train and test neural networks for student degree classification using different approaches:

#### Custom Neural Network (Project 7)
```bash
# Generate dataset
make student-dataset

# Train custom NN model
make train-student-model

# Test custom NN model
make test-student-model

# Or run complete pipeline
make student-model
```

#### TensorFlow/Keras (Project 9 - High Accuracy)
```bash
# Generate dataset (if not already done)
make student-dataset

# Train TensorFlow model (achieves >90% accuracy)
make train-tensorflow-student-model

# Test TensorFlow model
make test-tensorflow-student-model

# Or run complete TensorFlow pipeline
make tensorflow-student-model
```

**Model Comparison:**
- **Project 7 (Custom NN)**: ~62% accuracy, basic implementation
- **Project 8 (Advanced Custom NN)**: ~85% accuracy, feature engineering
- **Project 9 (TensorFlow/Keras)**: >90% accuracy, production-ready

The models classify students into 5 categories (Bad, Acceptable, Good, Very Good, Excellent) based on academic performance metrics.

## Features

- **Comprehensive Documentation**: Detailed markdown files with theory, code, and visualizations
- **Interactive Notebooks**: Jupyter notebooks for hands-on learning (11 notebooks)
- **Applied Projects**: 15+ real-world projects covering linear algebra and neural networks
- **Multiple Frameworks**: Custom neural networks and TensorFlow/Keras implementations
- **High Accuracy Models**: TensorFlow model achieves >90% accuracy on student classification
- **Automated Workflows**: Makefile for easy dataset generation, model training, and testing
- **Visual Learning**: Extensive use of graphs, charts, and Mermaid diagrams
- **Real Datasets**: Support for MNIST, Iris, and custom datasets
- **Modular Code**: Well-organized source code with visualization functions
- **Production Ready**: TensorFlow models can be saved and deployed

## Dataset Management

### Available Datasets

- **MNIST**: Handwritten digit recognition (generated on-demand, not stored in git)
- **Iris**: Flower classification dataset
- **Spam Detection**: Synthetic binary classification data
- **House Prices**: Synthetic regression dataset
- **XOR**: Non-linear classification problem
- **CNN Images**: Synthetic image classification data
- **Student Degree**: Academic performance classification (1000 students)

### Generating Datasets

```bash
# Generate all datasets
make datasets

# Generate specific dataset
make student-dataset

# Test dataset loading
make test-datasets
```

**Note**: Large dataset files (like MNIST) are excluded from git to avoid repository bloat. They can be regenerated using the scripts.

### Model Files

Trained models are saved in `data/neural_networks/`:
- `best_tensorflow_model.h5` - TensorFlow/Keras model (Project 9)
- `student_model_info.json` - Custom NN model metadata (Project 7)
- `tensorflow_model_info.json` - TensorFlow model metadata (Project 9)
- `training_history.csv` - Training history logs

Clean model files with: `make clean-models`

## Contributing

Feel free to add new learning sessions, projects, and examples!

## License

This project is for educational purposes.
