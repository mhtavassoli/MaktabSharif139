import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Set up the plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Parameters
np.random.seed(42)  # For reproducibility
n_samples = 1000
sigma_values = [0.5, 5, 20]

# Create subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Effect of Noise on Covariance and Correlation', fontsize=16, fontweight='bold')

# Initialize lists to store results
covariances = []
correlations = []

for i, sigma in enumerate(sigma_values):
    # Generate data according to the given relationship
    X = np.random.normal(0, 1, n_samples)
    noise = np.random.normal(0, sigma, n_samples)
    Y = 5 * X + noise
    
    # Calculate covariance and correlation
    covariance = np.cov(X, Y)[0, 1]
    correlation, _ = pearsonr(X, Y)
    
    covariances.append(covariance)
    correlations.append(correlation)
    
    # Plot scatter plots
    # Row 1: Scatter plots
    axes[i, 0].scatter(X, Y, alpha=0.6, s=20)
    axes[i, 0].set_xlabel('X')
    axes[i, 0].set_ylabel('Y')
    axes[i, 0].set_title(f'Scatter Plot (σ = {sigma})')
    
    # Add regression line
    z = np.polyfit(X, Y, 1)
    p = np.poly1d(z)
    axes[i, 0].plot(X, p(X), "r--", alpha=0.8, linewidth=2)
    
    # Row 2: X distribution
    axes[i, 1].hist(X, bins=30, alpha=0.7, density=True)
    axes[i, 1].set_xlabel('X')
    axes[i, 1].set_ylabel('Density')
    axes[i, 1].set_title(f'X Distribution (σ = {sigma})')
    
    # Row 3: Y distribution
    axes[i, 2].hist(Y, bins=30, alpha=0.7, density=True)
    axes[i, 2].set_xlabel('Y')
    axes[i, 2].set_ylabel('Density')
    axes[i, 2].set_title(f'Y Distribution (σ = {sigma})')

plt.tight_layout()
plt.show()

# Print numerical results
print("=" * 60)
print("NUMERICAL RESULTS")
print("=" * 60)
for i, sigma in enumerate(sigma_values):
    print(f"σ = {sigma}:")
    print(f"  Covariance: {covariances[i]:.4f}")
    print(f"  Correlation: {correlations[i]:.4f}")
    print(f"  Theoretical Covariance: {5:.4f}")
    print(f"  Theoretical Correlation: {5/np.sqrt(25 + sigma**2):.4f}")
    print("-" * 40)

# Additional analysis plot
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(sigma_values, covariances, 'bo-', markersize=8, linewidth=2)
plt.xlabel('Noise Standard Deviation (σ)')
plt.ylabel('Covariance')
plt.title('Covariance vs Noise Level')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(sigma_values, correlations, 'ro-', markersize=8, linewidth=2)
plt.xlabel('Noise Standard Deviation (σ)')
plt.ylabel('Correlation')
plt.title('Correlation vs Noise Level')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
