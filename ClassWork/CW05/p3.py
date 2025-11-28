import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualization
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# 1. Generate datasets
np.random.seed(42)  # For reproducible results
n_samples = 1000

# Normal distribution with σ=1, μ=0
normal_data = np.random.normal(0, 1, n_samples)

# Uniform distribution in range [-3, 3]
uniform_data = np.random.uniform(-3, 3, n_samples)

# 2. Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Normal distribution plots
# Histogram with bins=10
axes[0, 0].hist(normal_data, bins=10, alpha=0.7, density=True, edgecolor='black')
axes[0, 0].set_title('Normal Distribution\nHistogram (bins=10)')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Density')

# Histogram with bins=50
axes[0, 1].hist(normal_data, bins=50, alpha=0.7, density=True, edgecolor='black')
axes[0, 1].set_title('Normal Distribution\nHistogram (bins=50)')
axes[0, 1].set_xlabel('Value')
axes[0, 1].set_ylabel('Density')

# Boxplot
axes[0, 2].boxplot(normal_data)
axes[0, 2].set_title('Normal Distribution\nBoxplot')
axes[0, 2].set_ylabel('Value')

# Uniform distribution plots
# Histogram with bins=10
axes[1, 0].hist(uniform_data, bins=10, alpha=0.7, density=True, edgecolor='black')
axes[1, 0].set_title('Uniform Distribution\nHistogram (bins=10)')
axes[1, 0].set_xlabel('Value')
axes[1, 0].set_ylabel('Density')

# Histogram with bins=50
axes[1, 1].hist(uniform_data, bins=50, alpha=0.7, density=True, edgecolor='black')
axes[1, 1].set_title('Uniform Distribution\nHistogram (bins=50)')
axes[1, 1].set_xlabel('Value')
axes[1, 1].set_ylabel('Density')

# Boxplot
axes[1, 2].boxplot(uniform_data)
axes[1, 2].set_title('Uniform Distribution\nBoxplot')
axes[1, 2].set_ylabel('Value')

plt.tight_layout()
plt.show()

# Additional comparison plot
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(normal_data, bins=30, alpha=0.7, density=True, label='Normal', edgecolor='black')
plt.hist(uniform_data, bins=30, alpha=0.7, density=True, label='Uniform', edgecolor='black')
plt.title('Comparison: Normal vs Uniform Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

plt.subplot(1, 2, 2)
plt.boxplot([normal_data, uniform_data], labels=['Normal', 'Uniform'])
plt.title('Boxplot Comparison')
plt.ylabel('Value')

plt.tight_layout()
plt.show()

# Print basic statistics
print("Normal Distribution Statistics:")
print(f"Mean: {np.mean(normal_data):.3f}")
print(f"Standard Deviation: {np.std(normal_data):.3f}")
print(f"Min: {np.min(normal_data):.3f}, Max: {np.max(normal_data):.3f}")
print(f"Q1: {np.percentile(normal_data, 25):.3f}, Q3: {np.percentile(normal_data, 75):.3f}")

print("\nUniform Distribution Statistics:")
print(f"Mean: {np.mean(uniform_data):.3f}")
print(f"Standard Deviation: {np.std(uniform_data):.3f}")
print(f"Min: {np.min(uniform_data):.3f}, Max: {np.max(uniform_data):.3f}")
print(f"Q1: {np.percentile(uniform_data, 25):.3f}, Q3: {np.percentile(uniform_data, 75):.3f}")
