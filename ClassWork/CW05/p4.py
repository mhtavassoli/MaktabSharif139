import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =============================================================================
# 1 - Create dataset with N=10000 samples from uniform distribution [0,10]
# =============================================================================
np.random.seed(42)  # For reproducible results
N = 10000
dataset = np.random.uniform(0, 10, N)

print(f"Dataset created with {len(dataset)} samples")
print(f"Dataset mean: {np.mean(dataset):.3f}")
print(f"Dataset standard deviation: {np.std(dataset):.3f}")

# =============================================================================
# 2-4 - Loop 500 times, each time taking n=30 samples and calculating mean
# =============================================================================
num_iterations = 500
sample_size = 30
sample_means = []

for i in range(num_iterations):
    # Random sample of n=30 from dataset
    sample = np.random.choice(dataset, size=sample_size, replace=False)
    # Calculate mean and store
    sample_mean = np.mean(sample)
    sample_means.append(sample_mean)

sample_means = np.array(sample_means)

print(f"\nCollected {len(sample_means)} sample means")
print(f"Sample means mean: {np.mean(sample_means):.3f}")
print(f"Sample means standard deviation: {np.std(sample_means):.3f}")

# =============================================================================
# 5 - Analysis and Visualization for n=30
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Original dataset histogram
ax1.hist(dataset, bins=30, density=True, alpha=0.7, color='skyblue')
ax1.set_xlabel('Values')
ax1.set_ylabel('Density')
ax1.set_title('Original Uniform Distribution [0,10]\nN=10,000 samples')
ax1.grid(True, alpha=0.3)

# Plot 2: Sample means histogram for n=30
ax2.hist(sample_means, bins=30, density=True, alpha=0.7, color='lightcoral')
ax2.set_xlabel('Sample Means')
ax2.set_ylabel('Density')
ax2.set_title(f'Distribution of Sample Means\nn=30, {num_iterations} iterations')
ax2.grid(True, alpha=0.3)

# Add normal distribution curve for comparison
xmin, xmax = ax2.get_xlim()
x = np.linspace(xmin, xmax, 100)
theoretical_std = np.std(dataset) / np.sqrt(sample_size)
p = stats.norm.pdf(x, np.mean(dataset), theoretical_std)
ax2.plot(x, p, 'k', linewidth=2, label='Theoretical Normal Distribution')
ax2.legend()

plt.tight_layout()
plt.show()

# Statistical tests for normality
shapiro_stat, shapiro_p = stats.shapiro(sample_means)
print(f"\nShapiro-Wilk test for normality:")
print(f"Test statistic: {shapiro_stat:.4f}")
print(f"P-value: {shapiro_p:.4f}")

# =============================================================================
# 6 - Repeat with smaller sample size n=2
# =============================================================================
sample_size_small = 2
sample_means_small = []

for i in range(num_iterations):
    sample = np.random.choice(dataset, size=sample_size_small, replace=False)
    sample_mean = np.mean(sample)
    sample_means_small.append(sample_mean)

sample_means_small = np.array(sample_means_small)

# Plot comparison between n=30 and n=2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# n=30 case
ax1.hist(sample_means, bins=30, density=True, alpha=0.7, color='lightcoral')
xmin, xmax = ax1.get_xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, np.mean(dataset), theoretical_std)
ax1.plot(x, p, 'k', linewidth=2, label='Theoretical Normal')
ax1.set_xlabel('Sample Means')
ax1.set_ylabel('Density')
ax1.set_title(f'Distribution of Sample Means (n=30)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# n=2 case
ax2.hist(sample_means_small, bins=30, density=True, alpha=0.7, color='lightgreen')
theoretical_std_small = np.std(dataset) / np.sqrt(sample_size_small)
xmin, xmax = ax2.get_xlim()
x = np.linspace(xmin, xmax, 100)
p_small = stats.norm.pdf(x, np.mean(dataset), theoretical_std_small)
ax2.plot(x, p_small, 'k', linewidth=2, label='Theoretical Normal')
ax2.set_xlabel('Sample Means')
ax2.set_ylabel('Density')
ax2.set_title(f'Distribution of Sample Means (n=2)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nComparison of results:")
print(f"{'Sample Size':<12} {'Mean of Means':<15} {'Std of Means':<15}")
print(f"{'-' * 45}")
print(f"{'n=30':<12} {np.mean(sample_means):<15.3f} {np.std(sample_means):<15.3f}")
print(f"{'n=2':<12} {np.mean(sample_means_small):<15.3f} {np.std(sample_means_small):<15.3f}")
print(f"{'Theoretical':<12} {np.mean(dataset):<15.3f} {theoretical_std:<15.3f}")
