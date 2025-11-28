import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# 1. Create dataset with 500 samples from normal distribution
mean = 50
std_dev = 10
n_samples = 500

# Generate data from normal distribution with mean=50 and std=10
full_dataset = np.random.normal(mean, std_dev, n_samples)

print("Dataset created successfully!")
print(f"Dataset shape: {full_dataset.shape}")

# 2. Perform two types of sampling
# Complete sampling: using all 500 data points
complete_sample = full_dataset

# Incomplete sampling: random selection of only 30 data points
incomplete_sample = np.random.choice(full_dataset, size=30, replace=False)

print(f"\nComplete sample size: {len(complete_sample)}")
print(f"Incomplete sample size: {len(incomplete_sample)}")

# 3. Calculate and compare statistics for both cases
def calculate_statistics(data, sample_name):
    """
    Calculate mean, variance and standard deviation for given data
    
    Parameters:
    data: array of numerical values
    sample_name: name of the sample for printing
    """
    mean_val = np.mean(data)
    variance_val = np.var(data, ddof=1)  # Sample variance (unbiased estimator)
    std_dev_val = np.std(data, ddof=1)   # Sample standard deviation
    
    print(f"\n{sample_name} Statistics:")
    print(f"Mean: {mean_val:.4f}")
    print(f"Variance: {variance_val:.4f}")
    print(f"Standard Deviation: {std_dev_val:.4f}")
    
    return mean_val, variance_val, std_dev_val

# Calculate statistics for complete sample
complete_mean, complete_var, complete_std = calculate_statistics(complete_sample, "Complete Sampling")

# Calculate statistics for incomplete sample
incomplete_mean, incomplete_var, incomplete_std = calculate_statistics(incomplete_sample, "Incomplete Sampling")

# Compare the differences between complete and incomplete sampling
print("\n" + "="*50)
print("COMPARISON RESULTS:")
print("="*50)
print(f"Mean Difference: {abs(complete_mean - incomplete_mean):.4f}")
print(f"Variance Difference: {abs(complete_var - incomplete_var):.4f}")
print(f"Standard Deviation Difference: {abs(complete_std - incomplete_std):.4f}")

# Demonstrate variability with multiple small samples
print("\n" + "="*50)
print("DEMONSTRATING VARIABILITY WITH MULTIPLE SMALL SAMPLES:")
print("="*50)

# Take 10 different small samples to show variability
small_sample_means = []
small_sample_variances = []

for i in range(10):
    small_sample = np.random.choice(full_dataset, size=30, replace=False)
    sample_mean = np.mean(small_sample)
    sample_var = np.var(small_sample, ddof=1)
    small_sample_means.append(sample_mean)
    small_sample_variances.append(sample_var)
    print(f"Small sample {i+1}: Mean = {sample_mean:.4f}, Variance = {sample_var:.4f}")

# Calculate variability of small sample statistics
print(f"\nStandard deviation of small sample means: {np.std(small_sample_means):.4f}")
print(f"Standard deviation of small sample variances: {np.std(small_sample_variances):.4f}")