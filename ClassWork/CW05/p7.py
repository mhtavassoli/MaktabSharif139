import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# 1. Create datasets
N = 100

# Dataset A: Normal distribution with mean 0 and variance 2
A = np.random.normal(0, np.sqrt(2), N)

# Dataset B: Normal distribution with mean 0 and variance 2 + 10 outliers
B = np.random.normal(0, np.sqrt(2), N)
# Replace first 10 values with outliers
B[:10] = [-20, 20, -20, 20, -20, 20, -20, 20, -20, 20]

# 2. Calculate statistics
# For dataset A
range_A = np.max(A) - np.min(A)
std_dev_A = np.std(A)

# For dataset B
range_B = np.max(B) - np.min(B)
std_dev_B = np.std(B)

print("Dataset A Statistics:")
print(f"Range: {range_A:.4f}")
print(f"Standard Deviation: {std_dev_A:.4f}")
print()

print("Dataset B Statistics:")
print(f"Range: {range_B:.4f}")
print(f"Standard Deviation: {std_dev_B:.4f}")
print()

# 3. Visualization
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(A, bins=20, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(np.mean(A), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(A):.2f}')
plt.title('Dataset A: Normal Distribution (No Outliers)')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(B, bins=20, alpha=0.7, color='green', edgecolor='black')
plt.axvline(np.mean(B), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(B):.2f}')
plt.title('Dataset B: Normal Distribution + Outliers')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()

# Print comparison
print("Comparison:")
print(f"Range increase due to outliers: {((range_B - range_A) / range_A * 100):.2f}%")
print(f"Standard Deviation increase due to outliers: {((std_dev_B - std_dev_A) / std_dev_A * 100):.2f}%")
