# Documentation Index

Welcome to the comprehensive learning documentation for Mathematics and Deep Learning!

## 🗺️ [Learning Roadmap](LEARNING_ROADMAP.md)

**Start here if you're new!** The Learning Roadmap provides:
- Structured learning paths (Beginner → Intermediate → Advanced)
- Week-by-week study plans
- Timeline estimates (3.5-5 months full-time, 7-10 months part-time)
- Project recommendations
- Self-assessment checklists
- Custom learning paths for different goals (Data Scientists, Software Engineers, Researchers, Students)

[👉 Read the Learning Roadmap](LEARNING_ROADMAP.md)

## Mathematics Fundamentals

### [01. Linear Algebra](01_linear_algebra.md)
- Vectors and vector operations
- Matrices and matrix multiplication
- Eigenvalues and eigenvectors
- Linear transformations
- Visualizations with graphs

### [01a. Linear Algebra in Image Processing](01a_linear_algebra_image_processing.md) ⭐ NEW
- Image representation as matrices
- Image transformations (rotation, scaling, reflection)
- Image filtering and convolution
- Image compression using SVD
- Edge detection
- Image enhancement
- Color image processing

### [02. Calculus](02_calculus.md)
- Derivatives and tangent lines
- Numerical integration
- Chain rule

### [02a. Gradients](gradients.md) ⭐ NEW
- Gradient computation and visualization
- Gradient descent optimization
- Learning rate effects
- Neural network gradients
- Backpropagation and computation graphs

### [03. Statistics](03_statistics.md)
- Descriptive statistics
- Probability distributions
- Hypothesis testing
- Data normalization

## Deep Learning

### [04. Neural Networks](04_neural_networks.md)
- Perceptron and MLP architecture
- Forward and backward propagation
- Activation functions
- Training process visualization

### [05. Convolutional Neural Networks (CNNs)](05_cnns.md)
- Convolution operation
- Pooling layers
- CNN architecture
- Feature learning visualization

### [06. Recurrent Neural Networks (RNNs)](06_rnns.md)
- RNN architecture
- LSTM networks
- Sequence processing
- Applications

## Applied Projects

### [Linear Algebra Projects](projects/01_applied_linear_algebra_projects.md)
6 hands-on projects applying linear algebra:
1. **Image Transformations** - Rotate, scale, and reflect images using matrices
2. **Principal Component Analysis (PCA)** - Dimensionality reduction with eigenvalues
3. **Linear Regression from Scratch** - Implement using matrix operations
4. **Simple Neural Network** - Build a network using only matrix multiplication
5. **Data Preprocessing Pipeline** - Normalization and standardization
6. **Face Recognition with Eigenfaces** - PCA for face recognition

### [Neural Network Projects](projects/02_applied_neural_network_projects.md)
7 hands-on projects applying neural networks:
1. **Handwritten Digit Recognition (MNIST)** - Multi-class image classification
2. **Spam Detection** - Binary classification with neural networks
3. **Iris Classification** - Multi-class classification with decision boundaries
4. **House Price Prediction** - Regression with neural networks
5. **XOR Problem** - Demonstrating non-linearity importance
6. **CNN Image Classification** - Convolutional neural networks for images
7. **Student Degree Classification** - CSV-based multi-class classification

Each project has a corresponding Jupyter notebook in `notebooks/` for interactive learning!

## How to Use This Documentation

1. **Start with Mathematics**: Build a strong foundation in linear algebra, calculus, and statistics
2. **Progress to Neural Networks**: Understand the basics before moving to specialized architectures
3. **Run Code Examples**: Each document includes Python code that generates visualizations
4. **Practice Exercises**: Complete exercises at the end of each section
5. **Generate Graphs**: Run the code examples to create visualizations in `docs/images/`

## Running Code Examples

All documentation files include Python code examples. To run them:

```python
# Example: Run code from linear algebra documentation
exec(open('docs/01_linear_algebra.md').read())
```

Or copy the code blocks into Jupyter notebooks for interactive learning.

## Image Generation

All visualization code saves images to `docs/images/`. Make sure to run the code to generate:
- Vector and matrix visualizations
- Gradient descent plots
- Neural network architectures
- Training curves
- And more!

## Learning Path

**Beginner Path:**
1. Linear Algebra → Calculus → Statistics
2. Neural Networks → CNNs → RNNs
3. **Apply knowledge with [Projects](projects/01_applied_linear_algebra_projects.md)**

**Advanced Path:**
- Dive deeper into each topic
- Implement from scratch
- Experiment with different architectures
- Complete all applied projects
- Create your own projects

## Contributing

Feel free to add more examples, visualizations, or exercises to help students learn!
