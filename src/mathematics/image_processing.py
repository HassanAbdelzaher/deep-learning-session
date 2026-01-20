"""
Image Processing with Linear Algebra
Comprehensive module demonstrating linear algebra applications in image processing
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Some image processing functions may not work.")


# ============================================================================
# IMAGE CREATION
# ============================================================================

def create_sample_image(size=100):
    """Create a simple test image as a matrix"""
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Draw geometric shapes
    # Rectangle
    img[20:60, 20:60] = 255
    
    # Circle
    y, x = np.ogrid[:size, :size]
    center_x, center_y = 70, 70
    mask = (x - center_x)**2 + (y - center_y)**2 <= 20**2
    img[mask] = 180
    
    # Diagonal line
    for i in range(size):
        if 0 <= i < size:
            img[i, i] = 200
    
    return img

def create_complex_image():
    """Create image with more detail"""
    img = np.zeros((200, 200), dtype=np.uint8)
    
    # Add multiple shapes
    img[50:150, 50:150] = 255  # Large square
    img[20:80, 20:80] = 180    # Small square
    
    # Add gradient
    for i in range(200):
        img[i, :] = int(255 * i / 200)
    
    # Add circles
    y, x = np.ogrid[:200, :200]
    center_x, center_y = 100, 100
    mask = (x - center_x)**2 + (y - center_y)**2 <= 40**2
    img[mask] = 150
    
    return img

def create_color_image():
    """Create a color test image (3D matrix)"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Red channel
    img[20:60, 20:60, 0] = 255
    # Green channel
    img[40:80, 40:80, 1] = 255
    # Blue channel
    y, x = np.ogrid[:100, :100]
    center_x, center_y = 70, 30
    mask = (x - center_x)**2 + (y - center_y)**2 <= 15**2
    img[mask, 2] = 255
    
    return img


# ============================================================================
# IMAGE VISUALIZATION
# ============================================================================

def visualize_image_as_matrix(save_path: Optional[str] = None):
    """Visualize image representation as a matrix"""
    img = create_sample_image(100)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Image Visualization', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    im = axes[1].imshow(img, cmap='gray', aspect='auto')
    axes[1].set_title('Image as Matrix (Pixel Values)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Width (columns)', fontsize=12)
    axes[1].set_ylabel('Height (rows)', fontsize=12)
    plt.colorbar(im, ax=axes[1], label='Pixel Intensity')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ============================================================================
# IMAGE TRANSFORMATIONS
# ============================================================================

def apply_rotation_matrix(image, angle_degrees):
    """Rotate image using rotation matrix"""
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for image transformations")
    angle_rad = np.radians(angle_degrees)
    # Rotation matrix
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                  [np.sin(angle_rad), np.cos(angle_rad)]])
    
    # Apply rotation using scipy (handles interpolation)
    rotated = ndimage.rotate(image, angle_degrees, reshape=False, order=1)
    return rotated, R

def apply_scaling_matrix(image, scale_x, scale_y):
    """Scale image using scaling matrix"""
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for image transformations")
    S = np.array([[scale_x, 0],
                  [0, scale_y]])
    
    # Apply scaling
    scaled = ndimage.zoom(image, (scale_y, scale_x), order=1)
    # Crop to original size if needed
    h, w = image.shape
    if scaled.shape[0] > h or scaled.shape[1] > w:
        scaled = scaled[:h, :w]
    return scaled, S

def apply_reflection_matrix(image, axis='x'):
    """Reflect image using reflection matrix"""
    if axis == 'x':
        R = np.array([[1, 0], [0, -1]])  # Reflect across x-axis
        reflected = np.flipud(image)
    elif axis == 'y':
        R = np.array([[-1, 0], [0, 1]])  # Reflect across y-axis
        reflected = np.fliplr(image)
    else:  # y=x line
        R = np.array([[0, 1], [1, 0]])
        reflected = np.transpose(image)
    
    return reflected, R

def visualize_image_transformations(save_path: Optional[str] = None):
    """Visualize image transformations using matrix operations"""
    if not SCIPY_AVAILABLE:
        print("scipy required for image transformations")
        return None
    
    img = create_sample_image(100)
    
    # Define transformation matrices
    angle = 45
    angle_rad = np.radians(angle)
    R_rot = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                      [np.sin(angle_rad), np.cos(angle_rad)]])
    
    S_scale = np.array([[1.5, 0], [0, 1.5]])
    R_ref_x = np.array([[1, 0], [0, -1]])
    R_ref_y = np.array([[-1, 0], [0, 1]])
    
    # Apply transformations
    rotated, _ = apply_rotation_matrix(img, angle)
    scaled, _ = apply_scaling_matrix(img, 1.5, 1.5)
    reflected_x, _ = apply_reflection_matrix(img, 'x')
    reflected_y, _ = apply_reflection_matrix(img, 'y')
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    transformations = [
        ('Original', img, np.eye(2)),
        ('Rotation 45°', rotated, R_rot),
        ('Scaling 1.5x', scaled, S_scale),
        ('Reflection (x-axis)', reflected_x, R_ref_x),
        ('Reflection (y-axis)', reflected_y, R_ref_y),
    ]
    
    for idx, (name, transformed_img, T) in enumerate(transformations):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        ax.imshow(transformed_img, cmap='gray')
        det = np.linalg.det(T)
        ax.set_title(f'{name}\nDet = {det:.2f}', fontsize=11, fontweight='bold')
        ax.axis('off')
    
    axes[1, 2].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ============================================================================
# IMAGE FILTERING
# ============================================================================

def apply_convolution_filter(image, kernel):
    """Apply convolution filter to image (edge detection, blur, etc.)"""
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for image filtering")
    # Convolution is essentially matrix multiplication with a sliding window
    filtered = ndimage.convolve(image.astype(float), kernel, mode='constant')
    return filtered.astype(np.uint8)

def visualize_image_filtering(save_path: Optional[str] = None):
    """Visualize image filtering using convolution (matrix operations)"""
    if not SCIPY_AVAILABLE:
        print("scipy required for image filtering")
        return None
    
    img = create_sample_image(100)
    
    filters = {
        'Identity': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        'Edge Detection X': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        'Edge Detection Y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        'Blur': np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
        'Sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    }
    
    results = {}
    for name, kernel in filters.items():
        filtered = ndimage.convolve(img.astype(float), kernel, mode='constant')
        if filtered.min() < 0:
            filtered = filtered - filtered.min()
        if filtered.max() > 255:
            filtered = (filtered / filtered.max() * 255).astype(np.uint8)
        else:
            filtered = filtered.astype(np.uint8)
        results[name] = filtered
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original', fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    for idx, (name, filtered_img) in enumerate(results.items(), 1):
        axes[idx].imshow(filtered_img, cmap='gray')
        axes[idx].set_title(name, fontsize=11, fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ============================================================================
# IMAGE COMPRESSION
# ============================================================================

def compress_image_svd(image, k_components):
    """Compress image using SVD (Singular Value Decomposition)"""
    U, s, Vt = np.linalg.svd(image.astype(float), full_matrices=False)
    U_k = U[:, :k_components]
    s_k = s[:k_components]
    Vt_k = Vt[:k_components, :]
    compressed = U_k @ np.diag(s_k) @ Vt_k
    original_size = image.size
    compressed_size = U_k.size + s_k.size + Vt_k.size
    compression_ratio = original_size / compressed_size
    return compressed.astype(np.uint8), compression_ratio, s

def visualize_image_compression_svd(save_path: Optional[str] = None):
    """Visualize image compression using SVD (Singular Value Decomposition)"""
    img = create_complex_image()
    k_values = [5, 10, 20, 50, 100]
    compressions = {}
    
    for k in k_values:
        compressed, ratio, singular_values = compress_image_svd(img, k)
        compressions[k] = (compressed, ratio, singular_values)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f'Original\nSize: {img.size} pixels', fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    for idx, k in enumerate(k_values, 1):
        compressed, ratio, s = compressions[k]
        mse = np.mean((img.astype(float) - compressed.astype(float))**2)
        axes[idx].imshow(compressed, cmap='gray')
        axes[idx].set_title(f'k={k} components\nCompression: {ratio:.2f}x\nMSE: {mse:.1f}', 
                           fontsize=10, fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ============================================================================
# EDGE DETECTION
# ============================================================================

def edge_detection_sobel(image):
    """Edge detection using Sobel operators (matrix-based)"""
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for edge detection")
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    edges_x = ndimage.convolve(image.astype(float), sobel_x, mode='constant')
    edges_y = ndimage.convolve(image.astype(float), sobel_y, mode='constant')
    edges_mag = np.sqrt(edges_x**2 + edges_y**2)
    edges_dir = np.arctan2(edges_y, edges_x)
    
    return edges_x, edges_y, edges_mag, edges_dir

def visualize_edge_detection(save_path: Optional[str] = None):
    """Visualize edge detection using matrix operations"""
    if not SCIPY_AVAILABLE:
        print("scipy required for edge detection")
        return None
    
    img = create_complex_image()
    edges_x, edges_y, edges_mag, edges_dir = edge_detection_sobel(img)
    
    # Normalize for display
    edges_x_norm = ((edges_x - edges_x.min()) / (edges_x.max() - edges_x.min()) * 255).astype(np.uint8)
    edges_y_norm = ((edges_y - edges_y.min()) / (edges_y.max() - edges_y.min()) * 255).astype(np.uint8)
    edges_mag_norm = ((edges_mag - edges_mag.min()) / (edges_mag.max() - edges_mag.min()) * 255).astype(np.uint8)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(edges_x_norm, cmap='gray')
    axes[0, 1].set_title('Vertical Edges (Sobel X)\n∂I/∂x', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(edges_y_norm, cmap='gray')
    axes[1, 0].set_title('Horizontal Edges (Sobel Y)\n∂I/∂y', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(edges_mag_norm, cmap='gray')
    axes[1, 1].set_title('Edge Magnitude\n||∇I||', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ============================================================================
# IMAGE ENHANCEMENT
# ============================================================================

def enhance_brightness(image, factor):
    """Adjust brightness: I_new = I + factor (matrix addition)"""
    enhanced = np.clip(image.astype(float) + factor, 0, 255).astype(np.uint8)
    return enhanced

def enhance_contrast(image, factor):
    """Adjust contrast: I_new = factor × (I - mean) + mean (matrix scaling)"""
    mean = image.mean()
    enhanced = np.clip(factor * (image.astype(float) - mean) + mean, 0, 255).astype(np.uint8)
    return enhanced

def enhance_histogram_equalization(image):
    """Histogram equalization using cumulative distribution"""
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    enhanced = cdf_normalized[image].astype(np.uint8)
    return enhanced, hist, cdf


# ============================================================================
# COLOR IMAGE PROCESSING
# ============================================================================

def convert_to_grayscale(image):
    """Convert color to grayscale: Gray = 0.299×R + 0.587×G + 0.114×B (weighted sum)"""
    weights = np.array([0.299, 0.587, 0.114])
    grayscale = np.dot(image, weights).astype(np.uint8)
    return grayscale

def adjust_color_channel(image, channel, factor):
    """Adjust individual color channel (matrix scaling)"""
    adjusted = image.copy().astype(float)
    adjusted[:, :, channel] *= factor
    return np.clip(adjusted, 0, 255).astype(np.uint8)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=== Image Processing with Linear Algebra ===")
    
    if SCIPY_AVAILABLE:
        print("Generating image processing visualizations...")
        visualize_image_as_matrix('docs/images/image_as_matrix.png')
        visualize_image_transformations('docs/images/image_transformations_linear_algebra.png')
        visualize_image_filtering('docs/images/image_filtering_convolution.png')
        visualize_image_compression_svd('docs/images/image_compression_svd.png')
        visualize_edge_detection('docs/images/edge_detection_matrix_operations.png')
        print("Image processing visualizations complete!")
    else:
        print("Install scipy to enable image processing functions: pip install scipy")
        print("Basic functions (SVD compression) are still available.")
