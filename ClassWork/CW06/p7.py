# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

# Load the wine dataset
wine = load_wine()
X = wine.data
feature_names = wine.feature_names

# Create a DataFrame with numerical features
df = pd.DataFrame(X, columns=feature_names)
print(f"Original dataset shape: {df.shape}")
print(f"Features: {feature_names}")

# Calculate mean of each feature
feature_means = df.mean()

# Create an artificial outlier point (8 times the mean of each feature)
outlier_point = 8 * feature_means.values

# Add the outlier to the dataset
df_with_outlier = df.copy()
df_with_outlier.loc[len(df_with_outlier)] = outlier_point

print(f"\nDataset shape after adding outlier: {df_with_outlier.shape}")
print(f"Added outlier point: {outlier_point}")

# Standardize the data (important for Mahalanobis distance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_with_outlier)

# Calculate mean vector and covariance matrix
mean_vector = np.mean(X_scaled, axis=0)
cov_matrix = np.cov(X_scaled, rowvar=False)

# Calculate inverse of covariance matrix
try:
    inv_cov_matrix = np.linalg.inv(cov_matrix)
except np.linalg.LinAlgError:
    # If matrix is singular, use pseudo-inverse
    inv_cov_matrix = np.linalg.pinv(cov_matrix)

# Function to calculate Mahalanobis distance
def mahalanobis_distance(x, mean, inv_cov):
    diff = x - mean
    return np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))

# Calculate Mahalanobis distance for all points
mahalanobis_distances = np.array([mahalanobis_distance(x, mean_vector, inv_cov_matrix) for x in X_scaled])

# Calculate threshold based on 3 standard deviations
threshold = np.mean(mahalanobis_distances) + 3 * np.std(mahalanobis_distances)

# Alternatively, use chi-squared distribution threshold (for multivariate normal)
# degrees_of_freedom = X_scaled.shape[1]
# chi2_threshold = np.sqrt(chi2.ppf(0.975, degrees_of_freedom))

# Identify outliers
outlier_indices = np.where(mahalanobis_distances > threshold)[0]
outlier_labels = np.zeros(len(df_with_outlier))
outlier_labels[outlier_indices] = 1

print(f"\nMahalanobis distance statistics:")
print(f"Mean distance: {np.mean(mahalanobis_distances):.4f}")
print(f"Standard deviation: {np.std(mahalanobis_distances):.4f}")
print(f"Threshold (mean + 3σ): {threshold:.4f}")
print(f"\nNumber of outliers detected: {len(outlier_indices)}")
print(f"Outlier indices: {outlier_indices}")

# Check if our artificial outlier is detected
if len(df_with_outlier) - 1 in outlier_indices:
    print("✓ Artificial outlier successfully detected!")
else:
    print("✗ Artificial outlier NOT detected!")

# Visualization
plt.figure(figsize=(12, 5))

# Plot 1: Mahalanobis distances
plt.subplot(1, 2, 1)
plt.scatter(range(len(mahalanobis_distances)), mahalanobis_distances, 
            c=outlier_labels, cmap='coolwarm', alpha=0.7)
plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold = {threshold:.2f}')
plt.xlabel('Sample Index')
plt.ylabel('Mahalanobis Distance')
plt.title('Mahalanobis Distances with Outlier Detection')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: First two features with outliers highlighted
plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:-1, 0], X_scaled[:-1, 1], alpha=0.5, label='Normal points')
plt.scatter(X_scaled[outlier_indices, 0], X_scaled[outlier_indices, 1], 
            color='red', s=100, label='Outliers', marker='x')
# Highlight the artificial outlier
plt.scatter(X_scaled[-1, 0], X_scaled[-1, 1], 
            color='darkred', s=150, label='Artificial outlier', marker='*')
plt.xlabel('Feature 1 (standardized)')
plt.ylabel('Feature 2 (standardized)')
plt.title('Feature Space with Outliers')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Compare with Euclidean distance for analysis
euclidean_distances = np.sqrt(np.sum((X_scaled - mean_vector)**2, axis=1))
euclidean_threshold = np.mean(euclidean_distances) + 3 * np.std(euclidean_distances)
euclidean_outliers = np.where(euclidean_distances > euclidean_threshold)[0]

print(f"\nComparison with Euclidean distance:")
print(f"Euclidean threshold: {euclidean_threshold:.4f}")
print(f"Outliers detected by Euclidean distance: {len(euclidean_outliers)}")
if len(df_with_outlier) - 1 in euclidean_outliers:
    print("✓ Euclidean detected the artificial outlier")
else:
    print("✗ Euclidean did NOT detect the artificial outlier")
    