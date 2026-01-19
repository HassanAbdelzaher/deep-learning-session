"""
Visualization utilities
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_training_history(history, metrics=['loss', 'accuracy'], figsize=(12, 4)):
    """
    Plot training history
    
    Args:
        history: Dictionary with training metrics
        metrics: List of metrics to plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        if metric in history:
            axes[i].plot(history[metric], label=f'Training {metric}')
        if f'val_{metric}' in history:
            axes[i].plot(history[f'val_{metric}'], label=f'Validation {metric}')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].set_title(f'{metric.capitalize()} Over Time')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(8, 6)):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        figsize: Figure size
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    return plt.gcf()
