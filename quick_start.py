"""
Quick start script to demonstrate the learning project
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("=" * 60)
    print("Python AI Learning Sessions - Quick Start")
    print("=" * 60)
    print("\nThis project contains learning modules for:")
    print("  1. Mathematics (Linear Algebra, Calculus, Statistics)")
    print("  2. Deep Learning (Neural Networks, CNNs, RNNs)")
    print("\nAvailable modules:")
    print("\nMathematics:")
    print("  - src.mathematics.linear_algebra")
    print("  - src.mathematics.calculus")
    print("  - src.mathematics.statistics")
    print("\nDeep Learning:")
    print("  - src.deep_learning.neural_networks")
    print("  - src.deep_learning.cnn")
    print("  - src.deep_learning.rnn")
    print("\nExample usage:")
    print("  from src.mathematics import linear_algebra")
    print("  linear_algebra.vector_operations_example()")
    print("\n  from src.deep_learning import neural_networks")
    print("  neural_networks.xor_problem_example()")
    print("\nFor interactive learning, use Jupyter notebooks:")
    print("  jupyter lab notebooks/01_mathematics_basics.ipynb")
    print("  jupyter lab notebooks/02_neural_networks.ipynb")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
