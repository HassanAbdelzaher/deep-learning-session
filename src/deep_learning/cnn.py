"""
Convolutional Neural Networks (CNNs)
Comprehensive module with visualizations and examples
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy import ndimage
from typing import Optional, Tuple
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch


class SimpleCNN(nn.Module):
    """A simple Convolutional Neural Network"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Convolutional block 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Convolutional block 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def convolution_operation_example():
    """Demonstrate basic convolution operation"""
    # Create a simple image (5x5)
    image = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25]
    ])
    
    # Create a simple filter (3x3)
    filter_kernel = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ])
    
    # Manual convolution
    def convolve2d(image, kernel):
        """Simple 2D convolution"""
        kernel = np.flipud(np.fliplr(kernel))  # Flip kernel
        output = np.zeros_like(image)
        pad_width = kernel.shape[0] // 2
        padded_image = np.pad(image, pad_width, mode='constant')
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                output[i, j] = np.sum(
                    padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel
                )
        return output
    
    result = convolve2d(image, filter_kernel)
    
    print("Original Image:")
    print(image)
    print("\nFilter Kernel:")
    print(filter_kernel)
    print("\nConvolution Result:")
    print(result)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(filter_kernel, cmap='gray')
    axes[1].set_title('Filter Kernel')
    axes[1].axis('off')
    
    axes[2].imshow(result, cmap='gray')
    axes[2].set_title('Convolution Result')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return image, filter_kernel, result


def pooling_operation_example():
    """Demonstrate pooling operations"""
    # Create a simple feature map
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
    
    print("Original Feature Map (8x8):")
    print(feature_map)
    print("\nMax Pooling Result (4x4):")
    print(max_pooled)
    print("\nAverage Pooling Result (4x4):")
    print(avg_pooled)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(feature_map, cmap='viridis')
    axes[0].set_title('Original Feature Map')
    axes[0].axis('off')
    
    axes[1].imshow(max_pooled, cmap='viridis')
    axes[1].set_title('Max Pooling')
    axes[1].axis('off')
    
    axes[2].imshow(avg_pooled, cmap='viridis')
    axes[2].set_title('Average Pooling')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return feature_map, max_pooled, avg_pooled


def cnn_training_example():
    """Demonstrate CNN training on synthetic data"""
    # Generate synthetic image data (28x28 grayscale images)
    np.random.seed(42)
    num_samples = 1000
    num_classes = 10
    
    # Create synthetic data
    X = np.random.rand(num_samples, 1, 28, 28).astype(np.float32)
    y = np.random.randint(0, num_classes, num_samples)
    
    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y).long()
    
    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = SimpleCNN(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 5
    loss_history = []
    
    print("Training CNN...")
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Plot training loss
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CNN Training Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return model, loss_history


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_convolution_step_by_step(image: np.ndarray, kernel: np.ndarray,
                                      save_path: Optional[str] = None):
    """Visualize convolution operation step by step"""
    result = ndimage.convolve(image, kernel, mode='constant')
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    im1 = axes[0].imshow(image, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(kernel, cmap='RdBu', vmin=-1, vmax=1)
    axes[1].set_title('Filter Kernel', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(result, cmap='gray')
    axes[2].set_title('Convolution Result', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def visualize_cnn_architecture(save_path: Optional[str] = None):
    """Visualize CNN architecture"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    layers = [
        ('Input\nImage\n28×28', 1, 4, 'lightblue'),
        ('Conv1\n32 filters\n3×3', 3, 4, 'lightgreen'),
        ('ReLU', 4.5, 4, 'yellow'),
        ('MaxPool\n2×2', 6, 4, 'orange'),
        ('Conv2\n64 filters\n3×3', 8, 4, 'lightgreen'),
        ('ReLU', 9.5, 4, 'yellow'),
        ('MaxPool\n2×2', 11, 4, 'orange'),
        ('Flatten', 12.5, 4, 'lightcoral'),
        ('FC1\n128', 14, 4, 'lightblue'),
        ('FC2\n10', 15.5, 4, 'lightcoral')
    ]
    
    for label, x, y, color in layers:
        if 'Conv' in label or 'FC' in label or 'Input' in label:
            rect = Rectangle((x-0.4, y-0.3), 0.8, 0.6, color=color, ec='black', linewidth=2)
            ax.add_patch(rect)
        else:
            circle = Circle((x, y), 0.25, color=color, ec='black', linewidth=2)
            ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold')
    
    for i in range(len(layers) - 1):
        x1, y1 = layers[i][1] + 0.4, layers[i][2]
        x2, y2 = layers[i+1][1] - 0.4, layers[i+1][2]
        ax.arrow(x1, y1, x2-x1, y2-y1, head_width=0.1, head_length=0.1,
                fc='black', ec='black', linewidth=1.5)
    
    ax.set_xlim(0, 17)
    ax.set_ylim(3, 5)
    ax.set_title('CNN Architecture', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def visualize_feature_maps(model, input_tensor, layer_name: str = 'conv1',
                           save_path: Optional[str] = None):
    """Visualize feature maps from a CNN layer"""
    model.eval()
    with torch.no_grad():
        if layer_name == 'conv1':
            features = model.conv1(input_tensor)
            features = torch.relu(features)
        elif layer_name == 'conv2':
            x = torch.relu(model.conv1(input_tensor))
            x = model.pool(x)
            features = model.conv2(x)
            features = torch.relu(features)
        else:
            features = input_tensor
    
    features = features[0].cpu().numpy()
    num_filters = min(16, features.shape[0])
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(num_filters):
        row, col = i // 4, i % 4
        axes[row, col].imshow(features[i], cmap='viridis')
        axes[row, col].set_title(f'Filter {i+1}', fontsize=8)
        axes[row, col].axis('off')
    
    for i in range(num_filters, 16):
        row, col = i // 4, i % 4
        axes[row, col].axis('off')
    
    plt.suptitle(f'Feature Maps from {layer_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def visualize_pooling_comparison(feature_map: np.ndarray, pool_size: int = 2,
                                save_path: Optional[str] = None):
    """Compare max and average pooling"""
    def max_pooling(fm, ps):
        h, w = fm.shape
        output = np.zeros((h // ps, w // ps))
        for i in range(0, h, ps):
            for j in range(0, w, ps):
                output[i // ps, j // ps] = np.max(fm[i:i+ps, j:j+ps])
        return output
    
    def avg_pooling(fm, ps):
        h, w = fm.shape
        output = np.zeros((h // ps, w // ps))
        for i in range(0, h, ps):
            for j in range(0, w, ps):
                output[i // ps, j // ps] = np.mean(fm[i:i+ps, j:j+ps])
        return output
    
    max_pooled = max_pooling(feature_map, pool_size)
    avg_pooled = avg_pooling(feature_map, pool_size)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(feature_map, cmap='viridis')
    axes[0].set_title('Original Feature Map', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(max_pooled, cmap='viridis')
    axes[1].set_title(f'Max Pooling ({pool_size}×{pool_size})', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(avg_pooled, cmap='viridis')
    axes[2].set_title(f'Average Pooling ({pool_size}×{pool_size})', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


if __name__ == "__main__":
    print("=== Convolution Operation ===")
    convolution_operation_example()
    
    print("\n=== Pooling Operations ===")
    pooling_operation_example()
    
    print("\n=== CNN Training ===")
    cnn_training_example()
    
    print("\n=== Generating Visualizations ===")
    print("Run individual visualization functions to see graphs!")