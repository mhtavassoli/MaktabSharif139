import numpy as np
import matplotlib.pyplot as plt

# Define the Gaussian function
def gaussian_function(x, mu, sigma):
    """
    Calculate Gaussian function value for given x, mean (mu), and standard deviation (sigma)
    
    Parameters:
    x: input value
    mu: mean of the distribution
    sigma: standard deviation of the distribution
    
    Returns:
    Gaussian function value
    """
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Parameters
mu = 0  # Mean
x_min, x_max = -10, 10  # Range for x
num_samples = 1000  # Number of samples

# Generate x values
x = np.linspace(x_min, x_max, num_samples)

# Different sigma values to test
sigma_values = [0.5, 1, 3]

# Create plot
plt.figure(figsize=(12, 8))

# Calculate and plot Gaussian for each sigma value
for sigma in sigma_values:
    y = gaussian_function(x, mu, sigma)
    plt.plot(x, y, label=f'σ = {sigma}', linewidth=2)

# Customize plot
plt.title('Gaussian Function with Different Standard Deviations', fontsize=16, fontweight='bold')
plt.xlabel('x', fontsize=14)
plt.ylabel('f(x)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xlim(x_min, x_max)
plt.ylim(0, 1.1)

# Add some annotations to explain the effect
plt.annotate('Small σ: Narrow peak\n(High concentration around mean)', 
             xy=(-1, 0.8), xytext=(-8, 0.9),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10, color='red')

plt.annotate('Large σ: Wide spread\n(High dispersion)', 
             xy=(3, 0.3), xytext=(5, 0.7),
             arrowprops=dict(arrowstyle='->', color='green'),
             fontsize=10, color='green')

plt.tight_layout()
plt.show()

# Additional analysis: Calculate and display some properties# Create subplots for better comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, sigma in enumerate(sigma_values):
    y = gaussian_function(x, mu, sigma)
    
    axes[i].plot(x, y, 'b-', linewidth=2)
    axes[i].fill_between(x, y, alpha=0.3)
    axes[i].set_title(f'Gaussian with σ = {sigma}', fontsize=14)
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('f(x)')
    axes[i].grid(True, alpha=0.3)
    axes[i].set_xlim(-5, 5)
    axes[i].set_ylim(0, 1.1)

plt.tight_layout()
plt.show()
print("Analysis of Gaussian Functions:")
print("=" * 40)
for sigma in sigma_values:
    # Calculate full width at half maximum (FWHM)
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    print(f"For σ = {sigma}:")
    print(f"  - FWHM (Full Width at Half Maximum): {fwhm:.3f}")
    print(f"  - Variance (σ²): {sigma**2:.3f}")
    print(f"  - 68% of data within: [{mu-sigma:.1f}, {mu+sigma:.1f}]")
    print()

    # Additional Visualization
