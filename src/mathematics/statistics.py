"""
Probability and Statistics
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


def descriptive_statistics_example():
    """Demonstrate descriptive statistics"""
    # Generate sample data
    np.random.seed(42)
    data = np.random.normal(100, 15, 1000)  # Mean=100, Std=15
    
    # Calculate statistics
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    variance = np.var(data)
    min_val = np.min(data)
    max_val = np.max(data)
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    
    print("Descriptive Statistics:")
    print(f"  Mean: {mean:.2f}")
    print(f"  Median: {median:.2f}")
    print(f"  Standard Deviation: {std:.2f}")
    print(f"  Variance: {variance:.2f}")
    print(f"  Min: {min_val:.2f}")
    print(f"  Max: {max_val:.2f}")
    print(f"  25th Percentile: {q25:.2f}")
    print(f"  75th Percentile: {q75:.2f}")
    print(f"  IQR: {iqr:.2f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(data, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}')
    axes[0].axvline(median, color='g', linestyle='--', label=f'Median: {median:.2f}')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Histogram')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].boxplot(data, vert=True)
    axes[1].set_ylabel('Value')
    axes[1].set_title('Box Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'mean': mean, 'median': median, 'std': std,
        'variance': variance, 'min': min_val, 'max': max_val
    }


def probability_distributions_example():
    """Demonstrate common probability distributions"""
    x = np.linspace(-5, 5, 1000)
    
    # Normal distribution
    normal = stats.norm.pdf(x, loc=0, scale=1)
    
    # Uniform distribution
    uniform = stats.uniform.pdf(x, loc=-2, scale=4)
    
    # Exponential distribution
    x_exp = np.linspace(0, 5, 1000)
    exponential = stats.expon.pdf(x_exp, scale=1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(x, normal, 'b-', linewidth=2)
    axes[0].set_title('Normal Distribution (μ=0, σ=1)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('Probability Density')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(x, uniform, 'g-', linewidth=2)
    axes[1].set_title('Uniform Distribution (a=-2, b=2)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Probability Density')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(x_exp, exponential, 'r-', linewidth=2)
    axes[2].set_title('Exponential Distribution (λ=1)')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('Probability Density')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def correlation_example():
    """Demonstrate correlation analysis"""
    np.random.seed(42)
    n = 100
    
    # Create correlated data
    x = np.random.randn(n)
    y = 2 * x + np.random.randn(n) * 0.5  # Strong positive correlation
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(x, y)[0, 1]
    
    print(f"Correlation coefficient: {correlation:.3f}")
    
    # Visualize
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.6)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Scatter Plot (Correlation: {correlation:.3f})')
    plt.grid(True, alpha=0.3)
    
    # Add regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--", alpha=0.8, label=f'Regression: y={z[0]:.2f}x+{z[1]:.2f}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return correlation


def hypothesis_testing_example():
    """Demonstrate hypothesis testing"""
    np.random.seed(42)
    
    # Generate two samples
    sample1 = np.random.normal(100, 15, 100)
    sample2 = np.random.normal(105, 15, 100)
    
    # Perform t-test
    t_statistic, p_value = stats.ttest_ind(sample1, sample2)
    
    print("Hypothesis Testing (Two-sample t-test):")
    print(f"  Sample 1 mean: {np.mean(sample1):.2f}")
    print(f"  Sample 2 mean: {np.mean(sample2):.2f}")
    print(f"  t-statistic: {t_statistic:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significance level: 0.05")
    
    if p_value < 0.05:
        print("  Result: Reject null hypothesis (samples are significantly different)")
    else:
        print("  Result: Fail to reject null hypothesis (no significant difference)")
    
    return t_statistic, p_value


if __name__ == "__main__":
    print("=== Descriptive Statistics ===")
    descriptive_statistics_example()
    
    print("\n=== Probability Distributions ===")
    probability_distributions_example()
    
    print("\n=== Correlation Analysis ===")
    correlation_example()
    
    print("\n=== Hypothesis Testing ===")
    hypothesis_testing_example()
