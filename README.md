# Python AI Learning Sessions

A comprehensive Python project for learning mathematics and deep learning concepts through hands-on sessions, complete with visualizations, applied projects, and automated workflows.

## Project Structure

```
py-ai/
├── src/
│   ├── mathematics/          # Mathematical concepts and utilities
│   │   ├── linear_algebra.py
│   │   ├── calculus.py
│   │   └── statistics.py
│   ├── deep_learning/         # Deep learning implementations
│   │   ├── neural_networks.py
│   │   ├── cnn.py
│   │   └── rnn.py
│   └── utils/                 # Helper functions
│       ├── data_loader.py     # Dataset loading utilities
│       └── visualization.py
├── docs/                      # Comprehensive documentation with graphs
│   ├── 01_linear_algebra.md
│   ├── 02_calculus.md
│   ├── 03_statistics.md
│   ├── 04_neural_networks.md
│   ├── 05_cnns.md
│   ├── 06_rnns.md
│   ├── projects/              # Applied project documentation
│   └── images/                # Generated visualization images
├── notebooks/                 # Jupyter notebooks for interactive learning
│   ├── 01_mathematics_basics.ipynb
│   ├── 02_neural_networks.ipynb
│   └── 03-09_project_*.ipynb  # Applied project notebooks
├── scripts/                   # Automation scripts
│   ├── generate_neural_network_datasets.py
│   ├── generate_student_dataset.py
│   ├── train_student_model.py
│   └── test_student_model.py
├── data/                      # Datasets and data files
│   └── neural_networks/       # Generated datasets
├── tests/                     # Unit tests
├── sessions/                  # Learning session modules
├── Makefile                   # Automation commands
└── requirements.txt           # Python dependencies
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
- **[Applied Neural Network Projects](docs/projects/02_applied_neural_network_projects.md)** - 6 practical projects:
  - **Project 1: MNIST Digit Recognition** - Handwritten digit classification
  - **Project 2: Spam Detection** - Binary text classification
  - **Project 3: Iris Classification** - Multi-class flower classification
  - **Project 4: House Price Prediction** - Regression problem
  - **Project 5: XOR Problem** - Non-linearity demonstration
  - **Project 6: CNN Image Classification** - Convolutional neural networks
  - **Project 7: Student Degree Classification** - Real-world CSV data classification

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
- **09_project_7_student_degree_classification.ipynb** - Student degree classification from CSV

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

# Student Classification Model
make train-student-model   # Train the student degree classification model
make test-student-model    # Test the model on sample data
make student-model         # Complete pipeline (dataset + train + test)

# Development
make install              # Install dependencies
make install-dev          # Install with development tools
make lint                 # Run code linters
make format               # Format code with black
make test                 # Run all tests
make clean                # Clean generated files

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

### Student Classification Model

Train and test a neural network for student degree classification:

```bash
# Generate dataset
make student-dataset

# Train model
make train-student-model

# Test model
make test-student-model

# Or run complete pipeline
make student-model
```

The model classifies students into 5 categories (Bad, Acceptable, Good, Very Good, Excellent) based on academic performance metrics.

## Features

- **Comprehensive Documentation**: Detailed markdown files with theory, code, and visualizations
- **Interactive Notebooks**: Jupyter notebooks for hands-on learning
- **Applied Projects**: 13+ real-world projects covering linear algebra and neural networks
- **Automated Workflows**: Makefile for easy dataset generation, model training, and testing
- **Visual Learning**: Extensive use of graphs, charts, and Mermaid diagrams
- **Real Datasets**: Support for MNIST, Iris, and custom datasets
- **Modular Code**: Well-organized source code with visualization functions

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

## Contributing

Feel free to add new learning sessions, projects, and examples!

## License

This project is for educational purposes.
