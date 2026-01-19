from setuptools import setup, find_packages

setup(
    name="py-ai-learning",
    version="0.1.0",
    description="Python project for learning mathematics and deep learning",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "torch>=2.0.0",
        "tensorflow>=2.13.0",
        "jupyter>=1.0.0",
        "scikit-learn>=1.3.0",
    ],
    python_requires=">=3.8",
)
