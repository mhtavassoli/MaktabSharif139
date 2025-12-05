# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean, mahalanobis
import matplotlib.pyplot as plt

# Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names
class_names = wine.target_names

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Feature names: {feature_names}")
print(f"Class names: {class_names}")

# Standardize the data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Calculate class means in the original space
class_means_orig = []
for class_idx in range(3):
    class_data = X_std[y == class_idx]
    class_mean = np.mean(class_data, axis=0)
    class_means_orig.append(class_mean)
    
class_means_orig = np.array(class_means_orig)

# Function to calculate Euclidean distances
def calculate_euclidean_distances(means):
    distances = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i != j:
                distances[i, j] = euclidean(means[i], means[j])
    return distances

# Function to calculate Mahalanobis distances
def calculate_mahalanobis_distances(X, y, means):
    distances = np.zeros((3, 3))
    
    # Calculate covariance matrix for each class
    cov_matrices = []
    for class_idx in range(3):
        class_data = X[y == class_idx]
        # Use pseudo-inverse to handle potential singularity
        cov_matrix = np.cov(class_data, rowvar=False)
        cov_matrices.append(cov_matrix)
    
    for i in range(3):
        for j in range(3):
            if i != j:
                # For Mahalanobis distance, we need the covariance matrix
                # We'll use the pooled covariance or one of the class covariances
                # Here we use the covariance of class i
                try:
                    inv_cov = np.linalg.pinv(cov_matrices[i])
                    diff = means[i] - means[j]
                    distances[i, j] = np.sqrt(diff @ inv_cov @ diff.T)
                except:
                    # If calculation fails, use Euclidean as fallback
                    distances[i, j] = euclidean(means[i], means[j])
    
    return distances

# Calculate distances in original space
print("\n=== Original Space (13 dimensions) ===")
euc_dist_orig = calculate_euclidean_distances(class_means_orig)
print("Euclidean distances between class means:")
for i in range(3):
    for j in range(i+1, 3):
        print(f"  Class {i} - Class {j}: {euc_dist_orig[i, j]:.4f}")

mah_dist_orig = calculate_mahalanobis_distances(X_std, y, class_means_orig)
print("\nMahalanobis distances between class means:")
for i in range(3):
    for j in range(i+1, 3):
        print(f"  Class {i} - Class {j}: {mah_dist_orig[i, j]:.4f}")

# Apply PCA and reduce to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

print(f"\n=== PCA Results ===")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained by 2 components: {sum(pca.explained_variance_ratio_):.4f}")

# Calculate class means in PCA space
class_means_pca = []
for class_idx in range(3):
    class_data = X_pca[y == class_idx]
    class_mean = np.mean(class_data, axis=0)
    class_means_pca.append(class_mean)
    
class_means_pca = np.array(class_means_pca)

# Calculate distances in PCA space
print("\n=== PCA Space (2 dimensions) ===")
euc_dist_pca = calculate_euclidean_distances(class_means_pca)
print("Euclidean distances between class means:")
for i in range(3):
    for j in range(i+1, 3):
        print(f"  Class {i} - Class {j}: {euc_dist_pca[i, j]:.4f}")

mah_dist_pca = calculate_mahalanobis_distances(X_pca, y, class_means_pca)
print("\nMahalanobis distances between class means:")
for i in range(3):
    for j in range(i+1, 3):
        print(f"  Class {i} - Class {j}: {mah_dist_pca[i, j]:.4f}")

# Compare distance rankings
print("\n=== Comparison of Distance Rankings ===")
print("Euclidean distances - Original vs PCA:")
for i in range(3):
    for j in range(i+1, 3):
        orig = euc_dist_orig[i, j]
        pca_val = euc_dist_pca[i, j]
        print(f"  Class {i}-{j}: Original={orig:.4f}, PCA={pca_val:.4f}, Change={(pca_val-orig)/orig*100:.2f}%")

print("\nMahalanobis distances - Original vs PCA:")
for i in range(3):
    for j in range(i+1, 3):
        orig = mah_dist_orig[i, j]
        pca_val = mah_dist_pca[i, j]
        print(f"  Class {i}-{j}: Original={orig:.4f}, PCA={pca_val:.4f}, Change={(pca_val-orig)/orig*100:.2f}%")

# Visualize the results
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot in original space (first two features)
scatter1 = axes[0].scatter(X_std[:, 0], X_std[:, 1], c=y, cmap='viridis', alpha=0.6)
axes[0].scatter(class_means_orig[:, 0], class_means_orig[:, 1], c='red', s=200, marker='X', edgecolors='black')
for i in range(3):
    axes[0].annotate(f'Class {i}', (class_means_orig[i, 0], class_means_orig[i, 1]), 
                     fontsize=12, weight='bold')
axes[0].set_title('Original Space (First Two Features)')
axes[0].set_xlabel('Feature 1 (standardized)')
axes[0].set_ylabel('Feature 2 (standardized)')
axes[0].legend(handles=scatter1.legend_elements()[0], labels=class_names.tolist(), title='Classes')

# Plot in PCA space
scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
axes[1].scatter(class_means_pca[:, 0], class_means_pca[:, 1], c='red', s=200, marker='X', edgecolors='black')
for i in range(3):
    axes[1].annotate(f'Class {i}', (class_means_pca[i, 0], class_means_pca[i, 1]), 
                     fontsize=12, weight='bold')
axes[1].set_title('PCA Space (First Two Principal Components)')
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
axes[1].legend(handles=scatter2.legend_elements()[0], labels=class_names.tolist(), title='Classes')

plt.tight_layout()
plt.show()

# Print PCA loadings to understand what features are preserved
print("\n=== PCA Component Loadings ===")
print("Features with highest loadings on PC1:")
for i in np.argsort(np.abs(pca.components_[0]))[-5:][::-1]:
    print(f"  {feature_names[i]}: {pca.components_[0, i]:.4f}")

print("\nFeatures with highest loadings on PC2:")
for i in np.argsort(np.abs(pca.components_[1]))[-5:][::-1]:
    print(f"  {feature_names[i]}: {pca.components_[1, i]:.4f}")
    