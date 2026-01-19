# Statistics for Deep Learning

## Table of Contents
1. [Introduction](#introduction)
2. [Descriptive Statistics](#descriptive-statistics)
3. [Probability Distributions](#probability-distributions)
4. [Hypothesis Testing](#hypothesis-testing)
5. [Data Normalization](#data-normalization)

## Introduction

Statistics is crucial for:
- Understanding data distributions
- Normalizing inputs for neural networks
- Evaluating model performance
- Making predictions with uncertainty

## Descriptive Statistics

### Mean, Median, and Mode

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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
print(f"  25th Percentile (Q1): {q25:.2f}")
print(f"  75th Percentile (Q3): {q75:.2f}")
print(f"  Interquartile Range (IQR): {iqr:.2f}")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram
axes[0, 0].hist(data, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
axes[0, 0].axvline(mean, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
axes[0, 0].axvline(median, color='g', linestyle='--', linewidth=2, label=f'Median: {median:.2f}')
axes[0, 0].set_xlabel('Value', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title('Histogram with Mean and Median', fontsize=12, fontweight='bold')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# Box plot
bp = axes[0, 1].boxplot(data, vert=True, patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
axes[0, 1].set_ylabel('Value', fontsize=11)
axes[0, 1].set_title('Box Plot', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].text(1, q25, f'Q1: {q25:.1f}', fontsize=9, ha='center', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[0, 1].text(1, q75, f'Q3: {q75:.1f}', fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[0, 1].text(1, median, f'Median: {median:.1f}', fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Violin plot
axes[1, 0].violinplot([data], positions=[1], showmeans=True, showmedians=True)
axes[1, 0].set_ylabel('Value', fontsize=11)
axes[1, 0].set_title('Violin Plot (Distribution Shape)', fontsize=12, fontweight='bold')
axes[1, 0].set_xticks([1])
axes[1, 0].set_xticklabels(['Data'])
axes[1, 0].grid(True, alpha=0.3)

# Cumulative distribution
sorted_data = np.sort(data)
cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
axes[1, 1].plot(sorted_data, cumulative, linewidth=2, color='purple')
axes[1, 1].axhline(0.5, color='g', linestyle='--', alpha=0.7, label='50th percentile')
axes[1, 1].axvline(median, color='g', linestyle='--', alpha=0.7)
axes[1, 1].set_xlabel('Value', fontsize=11)
axes[1, 1].set_ylabel('Cumulative Probability', fontsize=11)
axes[1, 1].set_title('Cumulative Distribution Function (CDF)', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend(fontsize=9)

plt.tight_layout()
plt.savefig('docs/images/descriptive_statistics.png', dpi=150, bbox_inches='tight')
plt.show()
```

## Probability Distributions

### Normal Distribution

The normal (Gaussian) distribution is the most important in deep learning. Many assumptions rely on it.

```python
def visualize_normal_distribution():
    x = np.linspace(-5, 5, 1000)
    
    # Different parameters
    distributions = [
        (0, 1, 'μ=0, σ=1 (Standard Normal)', 'blue'),
        (0, 0.5, 'μ=0, σ=0.5', 'green'),
        (0, 2, 'μ=0, σ=2', 'red'),
        (2, 1, 'μ=2, σ=1', 'orange')
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # PDF (Probability Density Function)
    for mu, sigma, label, color in distributions:
        y = stats.norm.pdf(x, loc=mu, scale=sigma)
        axes[0].plot(x, y, label=label, linewidth=2, color=color)
    
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('Probability Density', fontsize=12)
    axes[0].set_title('Normal Distribution PDF', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # CDF (Cumulative Distribution Function)
    for mu, sigma, label, color in distributions:
        y = stats.norm.cdf(x, loc=mu, scale=sigma)
        axes[1].plot(x, y, label=label, linewidth=2, color=color)
    
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('Cumulative Probability', fontsize=12)
    axes[1].set_title('Normal Distribution CDF', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('docs/images/normal_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_normal_distribution()
```

### 68-95-99.7 Rule

For a normal distribution:
- 68% of data falls within 1 standard deviation
- 95% falls within 2 standard deviations
- 99.7% falls within 3 standard deviations

```python
def visualize_empirical_rule():
    mu, sigma = 0, 1
    x = np.linspace(-4, 4, 1000)
    y = stats.norm.pdf(x, mu, sigma)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot PDF
    ax.plot(x, y, 'b-', linewidth=2, label='Normal Distribution')
    
    # Shade regions
    x_1sigma = np.linspace(-1, 1, 100)
    y_1sigma = stats.norm.pdf(x_1sigma, mu, sigma)
    ax.fill_between(x_1sigma, 0, y_1sigma, alpha=0.3, color='green', label='68% (1σ)')
    
    x_2sigma = np.linspace(-2, 2, 100)
    y_2sigma = stats.norm.pdf(x_2sigma, mu, sigma)
    ax.fill_between(x_2sigma, 0, y_2sigma, alpha=0.2, color='orange', label='95% (2σ)')
    
    x_3sigma = np.linspace(-3, 3, 100)
    y_3sigma = stats.norm.pdf(x_3sigma, mu, sigma)
    ax.fill_between(x_3sigma, 0, y_3sigma, alpha=0.1, color='red', label='99.7% (3σ)')
    
    # Mark standard deviations
    for i in range(-3, 4):
        if i != 0:
            ax.axvline(i, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.text(i, 0.05, f'{i}σ', ha='center', fontsize=9)
    
    ax.axvline(0, color='black', linestyle='-', linewidth=1, label='Mean (μ)')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('68-95-99.7 Rule (Empirical Rule)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/images/empirical_rule.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_empirical_rule()
```

### Other Important Distributions

```python
def visualize_common_distributions():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Uniform distribution
    x_uniform = np.linspace(-1, 3, 1000)
    uniform = stats.uniform.pdf(x_uniform, loc=0, scale=2)
    axes[0, 0].plot(x_uniform, uniform, 'b-', linewidth=2)
    axes[0, 0].set_title('Uniform Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('x', fontsize=11)
    axes[0, 0].set_ylabel('PDF', fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Exponential distribution
    x_exp = np.linspace(0, 5, 1000)
    exp = stats.expon.pdf(x_exp, scale=1)
    axes[0, 1].plot(x_exp, exp, 'g-', linewidth=2)
    axes[0, 1].set_title('Exponential Distribution (λ=1)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('x', fontsize=11)
    axes[0, 1].set_ylabel('PDF', fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Binomial distribution
    n, p = 20, 0.5
    x_binom = np.arange(0, n+1)
    binom = stats.binom.pmf(x_binom, n, p)
    axes[1, 0].bar(x_binom, binom, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title(f'Binomial Distribution (n={n}, p={p})', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('x', fontsize=11)
    axes[1, 0].set_ylabel('PMF', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Poisson distribution
    lambda_poisson = 3
    x_poisson = np.arange(0, 15)
    poisson = stats.poisson.pmf(x_poisson, lambda_poisson)
    axes[1, 1].bar(x_poisson, poisson, color='red', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title(f'Poisson Distribution (λ={lambda_poisson})', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('x', fontsize=11)
    axes[1, 1].set_ylabel('PMF', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('docs/images/common_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_common_distributions()
```

## Hypothesis Testing

### T-Test Example

```python
def hypothesis_testing_example():
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
    print(f"  Significance level: α = 0.05")
    
    if p_value < 0.05:
        print("  Result: Reject null hypothesis (samples are significantly different)")
    else:
        print("  Result: Fail to reject null hypothesis (no significant difference)")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histograms
    axes[0].hist(sample1, bins=20, alpha=0.7, label=f'Sample 1 (μ={np.mean(sample1):.1f})', color='blue')
    axes[0].hist(sample2, bins=20, alpha=0.7, label=f'Sample 2 (μ={np.mean(sample2):.1f})', color='red')
    axes[0].axvline(np.mean(sample1), color='blue', linestyle='--', linewidth=2)
    axes[0].axvline(np.mean(sample2), color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Value', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Sample Distributions', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Box plot comparison
    axes[1].boxplot([sample1, sample2], labels=['Sample 1', 'Sample 2'], patch_artist=True)
    axes[1].set_ylabel('Value', fontsize=11)
    axes[1].set_title('Box Plot Comparison', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].text(1.5, axes[1].get_ylim()[1]*0.95, 
                f'p-value = {p_value:.4f}\n{"Significant" if p_value < 0.05 else "Not Significant"}',
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('docs/images/hypothesis_testing.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return t_statistic, p_value

hypothesis_testing_example()
```

## Data Normalization

### Why Normalize?

Neural networks work better when inputs are normalized. Common methods:

```python
def normalization_comparison():
    # Generate skewed data
    np.random.seed(42)
    data = np.random.exponential(scale=2, size=1000)
    
    # Standardization (Z-score normalization)
    mean = np.mean(data)
    std = np.std(data)
    standardized = (data - mean) / std
    
    # Min-Max normalization
    min_val = np.min(data)
    max_val = np.max(data)
    minmax_normalized = (data - min_val) / (max_val - min_val)
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Original data
    axes[0, 0].hist(data, bins=30, edgecolor='black', alpha=0.7, color='blue')
    axes[0, 0].set_title('Original Data', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Value', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].text(0.05, 0.95, f'Mean: {mean:.2f}\nStd: {std:.2f}', 
                   transform=axes[0, 0].transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[0, 0].grid(True, alpha=0.3)
    
    # Standardized
    axes[0, 1].hist(standardized, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_title('Standardized (Z-score)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Value', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].text(0.05, 0.95, f'Mean: {np.mean(standardized):.2f}\nStd: {np.std(standardized):.2f}', 
                   transform=axes[0, 1].transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[0, 1].grid(True, alpha=0.3)
    
    # Min-Max normalized
    axes[1, 0].hist(minmax_normalized, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].set_title('Min-Max Normalized', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Value', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].text(0.05, 0.95, f'Min: {np.min(minmax_normalized):.2f}\nMax: {np.max(minmax_normalized):.2f}', 
                   transform=axes[1, 0].transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 0].grid(True, alpha=0.3)
    
    # Comparison
    axes[1, 1].hist(data, bins=30, alpha=0.5, label='Original', color='blue')
    axes[1, 1].hist(standardized, bins=30, alpha=0.5, label='Standardized', color='green')
    axes[1, 1].hist(minmax_normalized, bins=30, alpha=0.5, label='Min-Max', color='orange')
    axes[1, 1].set_title('Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Value (normalized)', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('docs/images/normalization.png', dpi=150, bbox_inches='tight')
    plt.show()

normalization_comparison()
```

## Practice Exercises

1. **Descriptive Statistics**: Calculate mean, median, and standard deviation for: `[12, 15, 18, 20, 22, 25, 28, 30]`

2. **Normal Distribution**: What percentage of data falls within 2 standard deviations of the mean in a normal distribution?

3. **Normalization**: Normalize the data `[10, 20, 30, 40, 50]` using:
   - Standardization
   - Min-Max normalization

## Summary

- **Descriptive statistics** summarize data (mean, median, std, etc.)
- **Normal distribution** is fundamental in deep learning
- **68-95-99.7 rule** helps understand data spread
- **Hypothesis testing** evaluates statistical significance
- **Normalization** improves neural network training

## Next Steps

- Learn about [Neural Networks](04_neural_networks.md)
- Study [CNNs](05_cnns.md) for image processing
