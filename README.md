# Python AI Learning Sessions

A comprehensive Python project for learning mathematics and deep learning concepts through hands-on sessions.

## Project Structure

```
py-ai/
├── src/
│   ├── mathematics/          # Mathematical concepts and utilities
│   ├── deep_learning/        # Deep learning implementations
│   └── utils/                # Helper functions
├── docs/                     # Comprehensive documentation with graphs
│   ├── 01_linear_algebra.md
│   ├── 02_calculus.md
│   ├── 03_statistics.md
│   ├── 04_neural_networks.md
│   ├── 05_cnns.md
│   ├── 06_rnns.md
│   └── images/               # Generated visualization images
├── notebooks/                # Jupyter notebooks for interactive learning
├── data/                     # Datasets and data files
├── tests/                    # Unit tests
└── sessions/                 # Learning session modules
```

## Setup

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

4. **Start Jupyter Lab:**
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

Each documentation file includes:
- Detailed explanations
- Code examples with visualizations
- Graphs and diagrams
- Practice exercises
- Step-by-step tutorials

## Learning Sessions

### Mathematics Sessions
- Linear Algebra fundamentals
- Calculus and optimization
- Probability and statistics
- Numerical methods

### Deep Learning Sessions
- Neural Networks basics
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- Transformers and Attention mechanisms
- Generative Models

## Usage

### Using the Code Modules

Each session module can be run independently or imported as a library:

```python
from src.mathematics import linear_algebra
from src.deep_learning import neural_networks

# Run examples
linear_algebra.vector_operations_example()
neural_networks.xor_problem_example()
```

### Using the Documentation

1. **Read the markdown files** in `docs/` for comprehensive explanations
2. **Run code examples** from the documentation to generate visualizations
3. **Study the graphs** to understand concepts visually
4. **Complete exercises** at the end of each section

Example: Open `docs/01_linear_algebra.md` and run the Python code blocks to generate graphs showing vector operations, matrix multiplications, and more!

## Contributing

Feel free to add new learning sessions and examples!
