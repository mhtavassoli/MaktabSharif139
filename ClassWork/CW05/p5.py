import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
N = 1000
n1 = 500  # Number of samples from first distribution
n2 = 500  # Number of samples from second distribution

# Parameters for the two Gaussian distributions
mu1 = -2
sigma1 = 1
mu2 = 5
sigma2 = 1.5

# Generate samples from the two Gaussian distributions
samples1 = np.random.normal(mu1, sigma1, n1)
samples2 = np.random.normal(mu2, sigma2, n2)

# Combine the samples
combined_samples = np.concatenate([samples1, samples2])

# Calculate overall statistics
overall_mean = np.mean(combined_samples)
overall_variance = np.var(combined_samples)
overall_std = np.std(combined_samples)

# Calculate weighted mean of the component distributions
weighted_mean = (mu1 + mu2) / 2

print(f"Overall mean of combined dataset: {overall_mean:.4f}")
print(f"Overall variance of combined dataset: {overall_variance:.4f}")
print(f"Overall standard deviation of combined dataset: {overall_std:.4f}")
print(f"Weighted mean of component distributions: {weighted_mean:.4f}")

# Visualization
plt.figure(figsize=(12, 6))

# Plot histogram of combined dataset
plt.subplot(1, 2, 1)
plt.hist(combined_samples, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Histogram of Combined Gaussian Mixture')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

# Plot individual distributions for comparison
plt.subplot(1, 2, 2)
x = np.linspace(-6, 10, 1000)
pdf1 = (1/(sigma1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu1)/sigma1)**2)
pdf2 = (1/(sigma2 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu2)/sigma2)**2)
combined_pdf = 0.5 * pdf1 + 0.5 * pdf2

plt.plot(x, pdf1, 'r-', label=f'N(μ={mu1}, σ={sigma1})', linewidth=2)
plt.plot(x, pdf2, 'g-', label=f'N(μ={mu2}, σ={sigma2})', linewidth=2)
plt.plot(x, combined_pdf, 'b-', label='Combined Mixture', linewidth=2, alpha=0.7)
plt.title('Probability Density Functions')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional analysis
print("\n--- Additional Analysis ---")
print(f"Mean of first distribution: {mu1}")
print(f"Mean of second distribution: {mu2}")
print(f"Difference between overall mean and weighted mean: {abs(overall_mean - weighted_mean):.4f}")
