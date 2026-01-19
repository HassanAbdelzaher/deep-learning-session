# Python AI Learning Sessions

A comprehensive Python project for learning mathematics and deep learning concepts through hands-on sessions.

## Project Structure

```
py-ai/
├── src/
│   ├── mathematics/          # Mathematical concepts and utilities
│   ├── deep_learning/        # Deep learning implementations
│   └── utils/                # Helper functions
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

Each session module can be run independently or imported as a library:

```python
from src.mathematics import linear_algebra
from src.deep_learning import neural_networks
```

## Contributing

Feel free to add new learning sessions and examples!
