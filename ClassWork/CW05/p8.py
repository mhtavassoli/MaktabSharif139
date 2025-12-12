import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# 1 - Generate independent data
N = 500
mean = [0, 0]
cov1 = [[1, 0], [0, 1]]  # Identity matrix - no correlation

independent_data = np.random.multivariate_normal(mean, cov1, N)

# 2 - Generate correlated data
cov2 = [[1, 0.8], [0.8, 1]]  # Strong positive correlation

correlated_data = np.random.multivariate_normal(mean, cov2, N)

# 3 - Analysis and visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Scatter plot for independent data
ax1.scatter(independent_data[:, 0], independent_data[:, 1], alpha=0.6)
ax1.set_title('Independent Data ($Σ_1$)')
ax1.set_xlabel('$X_1$')
ax1.set_ylabel('$X_2$')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Scatter plot for correlated data
ax2.scatter(correlated_data[:, 0], correlated_data[:, 1], alpha=0.6, color='red')
ax2.set_title('Correlated Data ($Σ_2$)')
ax2.set_xlabel('$X_1$')
ax2.set_ylabel('$X_2$')
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

plt.tight_layout()
plt.show()

# Calculate empirical covariance matrices
empirical_cov1 = np.cov(independent_data.T)
empirical_cov2 = np.cov(correlated_data.T)

print("Theoretical covariance matrix Σ₁ :")
print(np.array(cov1))
print("\nEmpirical covariance matrix for independent data:")
print(empirical_cov1)

print("\nTheoretical covariance matrix Σ₂ :")
print(np.array(cov2))
print("\nEmpirical covariance matrix for correlated data:")
print(empirical_cov2)

# Calculate correlation coefficients
corr_coef1 = np.corrcoef(independent_data.T)[0, 1]
corr_coef2 = np.corrcoef(correlated_data.T)[0, 1]

print(f"\nCorrelation coefficient for independent data: {corr_coef1:.4f}")
print(f"Correlation coefficient for correlated data: {corr_coef2:.4f}")
