# Convolutional Neural Networks (CNNs)

## Table of Contents
1. [Introduction](#introduction)
2. [Convolution Operation](#convolution-operation)
3. [Pooling Layers](#pooling-layers)
4. [CNN Architecture](#cnn-architecture)
5. [Feature Visualization](#feature-visualization)
6. [Transfer Learning](#transfer-learning)

## Introduction

CNNs are specialized neural networks for processing grid-like data (images, time series). They use:
- **Convolutional layers**: Detect local patterns
- **Pooling layers**: Reduce dimensionality
- **Fully connected layers**: Make final predictions

## Convolution Operation

### What is Convolution?

Convolution applies a filter (kernel) to an image to detect features like edges, textures, etc.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def visualize_convolution():
    """Visualize convolution operation step by step"""
    # Create a simple 5x5 image
    image = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ])
    
    # Edge detection filter
    filter_kernel = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ])
    
    # Perform convolution
    result = ndimage.convolve(image, filter_kernel, mode='constant')
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    im1 = axes[0].imshow(image, cmap='gray', vmin=-1, vmax=1)
    axes[0].set_title('Input Image', fontsize=12, fontweight='bold')
    axes[0].set_xticks(range(5))
    axes[0].set_yticks(range(5))
    for i in range(5):
        for j in range(5):
            axes[0].text(j, i, int(image[i, j]), ha='center', va='center',
                       color='white' if image[i, j] > 0.5 else 'black', fontweight='bold')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(filter_kernel, cmap='RdBu', vmin=-1, vmax=1)
    axes[1].set_title('Filter Kernel\n(Edge Detection)', fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(3))
    axes[1].set_yticks(range(3))
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, int(filter_kernel[i, j]), ha='center', va='center',
                       color='white' if abs(filter_kernel[i, j]) > 0.5 else 'black', fontweight='bold')
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(result, cmap='RdBu', vmin=-3, vmax=3)
    axes[2].set_title('Convolution Result', fontsize=12, fontweight='bold')
    axes[2].set_xticks(range(5))
    axes[2].set_yticks(range(5))
    for i in range(5):
        for j in range(5):
            axes[2].text(j, i, int(result[i, j]), ha='center', va='center',
                       color='white' if abs(result[i, j]) > 1.5 else 'black', fontweight='bold')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('docs/images/convolution_operation.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_convolution()
```

### Common Filters

```python
def visualize_common_filters():
    """Show different types of filters"""
    # Create a test image with patterns
    image = np.zeros((9, 9))
    image[2:7, 2:7] = 1
    image[3:6, 3:6] = 0.5
    
    filters = {
        'Identity': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        'Edge Detection': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
        'Sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        'Blur': np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for idx, (name, kernel) in enumerate(filters.items()):
        result = ndimage.convolve(image, kernel, mode='constant')
        
        # Original
        if idx == 0:
            axes[0, idx].imshow(image, cmap='gray')
            axes[0, idx].set_title('Original Image', fontsize=11, fontweight='bold')
        else:
            axes[0, idx].axis('off')
        
        # Filter
        im = axes[1, idx].imshow(kernel, cmap='RdBu', vmin=-1, vmax=1)
        axes[1, idx].set_title(f'{name} Filter', fontsize=11, fontweight='bold')
        axes[1, idx].set_xticks(range(3))
        axes[1, idx].set_yticks(range(3))
        for i in range(3):
            for j in range(3):
                axes[1, idx].text(j, i, f'{kernel[i, j]:.2f}', ha='center', va='center',
                                fontsize=8, color='white' if abs(kernel[i, j]) > 0.5 else 'black')
        plt.colorbar(im, ax=axes[1, idx])
        
        # Result
        if idx < 4:
            im2 = axes[0, idx].imshow(result, cmap='gray')
            axes[0, idx].set_title(f'{name} Result', fontsize=11, fontweight='bold')
            plt.colorbar(im2, ax=axes[0, idx])
    
    plt.tight_layout()
    plt.savefig('docs/images/common_filters.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_common_filters()
```

## Pooling Layers

Pooling reduces spatial dimensions while retaining important information.

### Max Pooling

```python
def visualize_pooling():
    """Visualize max and average pooling"""
    # Create feature map
    feature_map = np.random.rand(8, 8) * 100
    
    # Max pooling
    def max_pooling(feature_map, pool_size=2):
        h, w = feature_map.shape
        output = np.zeros((h // pool_size, w // pool_size))
        for i in range(0, h, pool_size):
            for j in range(0, w, pool_size):
                output[i // pool_size, j // pool_size] = np.max(
                    feature_map[i:i+pool_size, j:j+pool_size]
                )
        return output
    
    # Average pooling
    def avg_pooling(feature_map, pool_size=2):
        h, w = feature_map.shape
        output = np.zeros((h // pool_size, w // pool_size))
        for i in range(0, h, pool_size):
            for j in range(0, w, pool_size):
                output[i // pool_size, j // pool_size] = np.mean(
                    feature_map[i:i+pool_size, j:j+pool_size]
                )
        return output
    
    max_pooled = max_pooling(feature_map)
    avg_pooled = avg_pooling(feature_map)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    im1 = axes[0].imshow(feature_map, cmap='viridis')
    axes[0].set_title('Original Feature Map (8×8)', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(max_pooled, cmap='viridis')
    axes[1].set_title('Max Pooling (4×4)', fontsize=12, fontweight='bold')
    for i in range(4):
        for j in range(4):
            axes[1].text(j, i, f'{max_pooled[i, j]:.1f}', ha='center', va='center',
                        fontsize=9, color='white' if max_pooled[i, j] > 50 else 'black')
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(avg_pooled, cmap='viridis')
    axes[2].set_title('Average Pooling (4×4)', fontsize=12, fontweight='bold')
    for i in range(4):
        for j in range(4):
            axes[2].text(j, i, f'{avg_pooled[i, j]:.1f}', ha='center', va='center',
                        fontsize=9, color='white' if avg_pooled[i, j] > 50 else 'black')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('docs/images/pooling_operations.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_pooling()
```

## CNN Architecture

### Typical CNN Structure

```python
def visualize_cnn_architecture():
    """Visualize a typical CNN architecture"""
    from matplotlib.patches import Rectangle, FancyBboxPatch
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Layer positions and sizes
    layers = [
        ('Input\n224×224×3', (1, 4), (2, 2), 'lightblue'),
        ('Conv1\n112×112×64', (4, 3.5), (2, 1.5), 'lightgreen'),
        ('Pool1\n56×56×64', (7, 3.5), (1.5, 1.5), 'lightyellow'),
        ('Conv2\n56×56×128', (9.5, 3.5), (2, 1.5), 'lightgreen'),
        ('Pool2\n28×28×128', (12.5, 3.5), (1.5, 1.5), 'lightyellow'),
        ('Conv3\n28×28×256', (15, 3.5), (2, 1.5), 'lightgreen'),
        ('Pool3\n14×14×256', (18, 3.5), (1.5, 1.5), 'lightyellow'),
        ('FC1\n1024', (20.5, 3.5), (1.5, 1.5), 'lightcoral'),
        ('FC2\n512', (23, 3.5), (1.5, 1.5), 'lightcoral'),
        ('Output\n10', (25.5, 3.5), (1.5, 1.5), 'orange')
    ]
    
    # Draw layers
    for name, (x, y), (w, h), color in layers:
        box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1',
                            facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, name, ha='center', va='center',
               fontsize=9, fontweight='bold')
    
    # Draw arrows
    for i in range(len(layers) - 1):
        x1 = layers[i][1][0] + layers[i][2][0]
        x2 = layers[i+1][1][0]
        y = layers[i][1][1] + layers[i][2][1] / 2
        arrow = FancyArrowPatch((x1, y), (x2, y), arrowstyle='->',
                               mutation_scale=20, linewidth=2, color='blue')
        ax.add_patch(arrow)
    
    ax.set_xlim(0, 27)
    ax.set_ylim(2, 6)
    ax.set_title('Typical CNN Architecture', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('docs/images/cnn_architecture.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_cnn_architecture()
```

## Feature Visualization

### How CNNs Learn Features

```python
def visualize_feature_learning():
    """Show how different layers detect different features"""
    # Simulate feature maps at different layers
    np.random.seed(42)
    
    # Early layers: edges, simple patterns
    early_layer = np.random.rand(32, 32)
    early_layer[10:15, :] = 0.8  # Horizontal edge
    early_layer[:, 20:25] = 0.8  # Vertical edge
    
    # Middle layers: textures, patterns
    middle_layer = np.random.rand(16, 16)
    for i in range(0, 16, 4):
        for j in range(0, 16, 4):
            middle_layer[i:i+2, j:j+2] = 0.9
    
    # Deep layers: complex features, objects
    deep_layer = np.random.rand(8, 8)
    deep_layer[2:6, 2:6] = 0.9  # Object-like pattern
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    im1 = axes[0].imshow(early_layer, cmap='hot')
    axes[0].set_title('Early Layers\n(Edges, Simple Patterns)', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(middle_layer, cmap='hot')
    axes[1].set_title('Middle Layers\n(Textures, Patterns)', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(deep_layer, cmap='hot')
    axes[2].set_title('Deep Layers\n(Complex Features, Objects)', fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('docs/images/feature_learning.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_feature_learning()
```

## Practice Exercises

1. **Convolution**: Apply a 3×3 edge detection filter to a 5×5 image manually.

2. **Pooling**: Perform 2×2 max pooling on a 4×4 feature map.

3. **Architecture**: Design a CNN for 32×32 RGB image classification with 10 classes.

## Summary

- **Convolution** detects local patterns using filters
- **Pooling** reduces dimensions while preserving important information
- **CNNs** stack conv and pooling layers to learn hierarchical features
- **Early layers** detect edges and simple patterns
- **Deep layers** detect complex features and objects

## Next Steps

- Learn about [RNNs](06_rnns.md) for sequence data
- Study [Transfer Learning](../docs/07_transfer_learning.md)
