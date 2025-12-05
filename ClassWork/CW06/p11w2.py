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
                # We'll use the covariance of class i
                try:
                    inv_cov = np.linalg.pinv(cov_matrices[i])
                    diff = means[i] - means[j]
                    distances[i, j] = np.sqrt(diff @ inv_cov @ diff.T)
                except:
                    # If calculation fails, use Euclidean as fallback
                    distances[i, j] = euclidean(means[i], means[j])
    
    return distances

# Calculate distances in original space
print("\n" + "="*50)
print("Original Space (13 dimensions)")
print("="*50)
euc_dist_orig = calculate_euclidean_distances(class_means_orig)
print("\nEuclidean distances between class means:")
for i in range(3):
    for j in range(i+1, 3):
        print(f"  Class {i} ({class_names[i]}) - Class {j} ({class_names[j]}): {euc_dist_orig[i, j]:.4f}")

mah_dist_orig = calculate_mahalanobis_distances(X_std, y, class_means_orig)
print("\nMahalanobis distances between class means:")
for i in range(3):
    for j in range(i+1, 3):
        print(f"  Class {i} ({class_names[i]}) - Class {j} ({class_names[j]}): {mah_dist_orig[i, j]:.4f}")

# Determine closest classes in original space
print("\nClosest classes in original space:")
print("Based on Euclidean distance:")
min_euc = np.min(euc_dist_orig[euc_dist_orig > 0])
min_indices = np.where(euc_dist_orig == min_euc)
for idx in zip(*min_indices):
    if idx[0] < idx[1]:
        print(f"  Classes {idx[0]} and {idx[1]} with distance: {min_euc:.4f}")

print("\nBased on Mahalanobis distance:")
min_mah = np.min(mah_dist_orig[mah_dist_orig > 0])
min_indices = np.where(mah_dist_orig == min_mah)
for idx in zip(*min_indices):
    if idx[0] < idx[1]:
        print(f"  Classes {idx[0]} and {idx[1]} with distance: {min_mah:.4f}")

# Apply PCA and reduce to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

print("\n" + "="*50)
print("PCA Results")
print("="*50)
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
print("\n" + "="*50)
print("PCA Space (2 dimensions)")
print("="*50)
euc_dist_pca = calculate_euclidean_distances(class_means_pca)
print("\nEuclidean distances between class means:")
for i in range(3):
    for j in range(i+1, 3):
        print(f"  Class {i} ({class_names[i]}) - Class {j} ({class_names[j]}): {euc_dist_pca[i, j]:.4f}")

mah_dist_pca = calculate_mahalanobis_distances(X_pca, y, class_means_pca)
print("\nMahalanobis distances between class means:")
for i in range(3):
    for j in range(i+1, 3):
        print(f"  Class {i} ({class_names[i]}) - Class {j} ({class_names[j]}): {mah_dist_pca[i, j]:.4f}")

# Determine closest classes in PCA space
print("\nClosest classes in PCA space:")
print("Based on Euclidean distance:")
min_euc_pca = np.min(euc_dist_pca[euc_dist_pca > 0])
min_indices = np.where(euc_dist_pca == min_euc_pca)
for idx in zip(*min_indices):
    if idx[0] < idx[1]:
        print(f"  Classes {idx[0]} and {idx[1]} with distance: {min_euc_pca:.4f}")

print("\nBased on Mahalanobis distance:")
min_mah_pca = np.min(mah_dist_pca[mah_dist_pca > 0])
min_indices = np.where(mah_dist_pca == min_mah_pca)
for idx in zip(*min_indices):
    if idx[0] < idx[1]:
        print(f"  Classes {idx[0]} and {idx[1]} with distance: {min_mah_pca:.4f}")

# Compare distance rankings
print("\n" + "="*50)
print("Comparison of Distance Rankings")
print("="*50)

print("\nEuclidean distances - Original vs PCA:")
print("Class Pair | Original | PCA | Change")
print("-" * 45)
for i in range(3):
    for j in range(i+1, 3):
        orig = euc_dist_orig[i, j]
        pca_val = euc_dist_pca[i, j]
        change = (pca_val - orig) / orig * 100
        print(f"{i}-{j} | {orig:.4f} | {pca_val:.4f} | {change:+.2f}%")

print("\nMahalanobis distances - Original vs PCA:")
print("Class Pair | Original | PCA | Change")
print("-" * 45)
for i in range(3):
    for j in range(i+1, 3):
        orig = mah_dist_orig[i, j]
        pca_val = mah_dist_pca[i, j]
        change = (pca_val - orig) / orig * 100
        print(f"{i}-{j} | {orig:.4f} | {pca_val:.4f} | {change:+.2f}%")

# Check if the order of closeness changed
print("\n" + "="*50)
print("Did the order of closeness change?")
print("="*50)

# For Euclidean distances
euc_order_orig = []
for i in range(3):
    for j in range(i+1, 3):
        euc_order_orig.append((i, j, euc_dist_orig[i, j]))

euc_order_orig_sorted = sorted(euc_order_orig, key=lambda x: x[2])
print("\nEuclidean - Original space order (closest to farthest):")
for idx, (i, j, dist) in enumerate(euc_order_orig_sorted):
    print(f"  {idx+1}. Classes {i}-{j}: {dist:.4f}")

euc_order_pca = []
for i in range(3):
    for j in range(i+1, 3):
        euc_order_pca.append((i, j, euc_dist_pca[i, j]))

euc_order_pca_sorted = sorted(euc_order_pca, key=lambda x: x[2])
print("\nEuclidean - PCA space order (closest to farthest):")
for idx, (i, j, dist) in enumerate(euc_order_pca_sorted):
    print(f"  {idx+1}. Classes {i}-{j}: {dist:.4f}")

# Compare if order changed
order_changed_euc = euc_order_orig_sorted[0][:2] != euc_order_pca_sorted[0][:2]
print(f"\nDid the closest pair change with Euclidean distance? {'YES' if order_changed_euc else 'NO'}")

# Visualize the results with corrected legend handling
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot in original space (first two features)
for class_idx in range(3):
    mask = (y == class_idx)
    axes[0].scatter(X_std[mask, 0], X_std[mask, 1], label=f'Class {class_idx} ({class_names[class_idx]})', alpha=0.6)

axes[0].scatter(class_means_orig[:, 0], class_means_orig[:, 1], c='red', s=200, marker='X', edgecolors='black', label='Class Means')
for i in range(3):
    axes[0].annotate(f'Mean {i}', (class_means_orig[i, 0], class_means_orig[i, 1]), 
                     fontsize=10, weight='bold', xytext=(5, 5), textcoords='offset points')
axes[0].set_title('Original Space (First Two Features)')
axes[0].set_xlabel('Feature 1 (Alcohol)')
axes[0].set_ylabel('Feature 2 (Malic Acid)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot in PCA space
for class_idx in range(3):
    mask = (y == class_idx)
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Class {class_idx} ({class_names[class_idx]})', alpha=0.6)

axes[1].scatter(class_means_pca[:, 0], class_means_pca[:, 1], c='red', s=200, marker='X', edgecolors='black', label='Class Means')
for i in range(3):
    axes[1].annotate(f'Mean {i}', (class_means_pca[i, 0], class_means_pca[i, 1]), 
                     fontsize=10, weight='bold', xytext=(5, 5), textcoords='offset points')
axes[1].set_title('PCA Space (First Two Principal Components)')
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print PCA loadings to understand what features are preserved
print("\n" + "="*50)
print("PCA Component Loadings")
print("="*50)
print("\nFeatures with highest absolute loadings on PC1:")
pc1_loadings = list(zip(feature_names, pca.components_[0]))
pc1_sorted = sorted(pc1_loadings, key=lambda x: abs(x[1]), reverse=True)[:5]
for feature, loading in pc1_sorted:
    print(f"  {feature}: {loading:.4f}")

print("\nFeatures with highest absolute loadings on PC2:")
pc2_loadings = list(zip(feature_names, pca.components_[1]))
pc2_sorted = sorted(pc2_loadings, key=lambda x: abs(x[1]), reverse=True)[:5]
for feature, loading in pc2_sorted:
    print(f"  {feature}: {loading:.4f}")

# Analyze what features are lost
print("\n" + "="*50)
print("Analysis: What features are effectively lost?")
print("="*50)
print("\nFeatures with lowest absolute loadings on both PC1 and PC2:")
all_loadings = []
for idx, feature in enumerate(feature_names):
    pc1_val = abs(pca.components_[0, idx])
    pc2_val = abs(pca.components_[1, idx])
    avg_loading = (pc1_val + pc2_val) / 2
    all_loadings.append((feature, avg_loading))

least_important = sorted(all_loadings, key=lambda x: x[1])[:5]
for feature, loading in least_important:
    print(f"  {feature}: average loading = {loading:.4f}")