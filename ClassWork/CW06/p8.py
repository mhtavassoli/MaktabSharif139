# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load dataset and convert to DataFrame
wine = load_wine()
df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
target = wine.target
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# 2. Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 3. Apply PCA on all features
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 4. Create eigenvalues and explained variance table
explained_variance = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

variance_df = pd.DataFrame({
    'PC_index': [f'PC{i+1}' for i in range(len(explained_variance))],
    'eigenvalue': explained_variance,
    'explained_variance_ratio': explained_variance_ratio,
    'cumulative_explained_variance': cumulative_explained_variance
})

print("\n" + "="*80)
print("EXPLAINED VARIANCE TABLE:")
print("="*80)
print(variance_df.to_string(index=False))

# 5. Find number of components for 90% variance
n_components_90 = np.argmax(cumulative_explained_variance >= 0.90) + 1
print(f"\nNumber of components needed for 90% variance: {n_components_90}")
print(f"Cumulative variance with {n_components_90} components: {cumulative_explained_variance[n_components_90-1]:.4f}")

# 6. Create loadings table (eigenvectors)
loadings = pca.components_.T  # Transpose to have features as rows
loadings_df = pd.DataFrame(
    loadings,
    index=wine.feature_names,
    columns=[f'PC{i+1}' for i in range(loadings.shape[1])]
)

print("\n" + "="*80)
print("LOADINGS TABLE (Eigenvectors):")
print("="*80)
print(loadings_df.round(4))

# 7. Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Scree plot
axes[0, 0].plot(range(1, len(explained_variance_ratio)+1), explained_variance_ratio, 'bo-', linewidth=2)
axes[0, 0].plot(range(1, len(explained_variance_ratio)+1), cumulative_explained_variance, 'ro-', linewidth=2)
axes[0, 0].axhline(y=0.90, color='g', linestyle='--', alpha=0.7)
axes[0, 0].set_title('Scree Plot and Cumulative Variance')
axes[0, 0].set_xlabel('Principal Components')
axes[0, 0].set_ylabel('Explained Variance Ratio')
axes[0, 0].legend(['Individual', 'Cumulative', '90% Threshold'])
axes[0, 0].grid(True, alpha=0.3)

# Loadings heatmap for first 5 PCs
sns.heatmap(loadings_df.iloc[:, :5], annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, ax=axes[0, 1])
axes[0, 1].set_title('Loadings Heatmap (First 5 Principal Components)')

# First two principal components colored by target
scatter = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=target, cmap='viridis', alpha=0.7)
axes[1, 0].set_xlabel(f'PC1 ({explained_variance_ratio[0]:.2%})')
axes[1, 0].set_ylabel(f'PC2 ({explained_variance_ratio[1]:.2%})')
axes[1, 0].set_title('PCA: PC1 vs PC2 Colored by Wine Class')
plt.colorbar(scatter, ax=axes[1, 0])

# Bar plot of loadings for PC1
loadings_pc1 = loadings_df['PC1'].sort_values()
axes[1, 1].barh(range(len(loadings_pc1)), loadings_pc1.values)
axes[1, 1].set_yticks(range(len(loadings_pc1)))
axes[1, 1].set_yticklabels(loadings_pc1.index)
axes[1, 1].set_xlabel('Loading Value')
axes[1, 1].set_title('Feature Loadings on PC1')
axes[1, 1].axvline(x=0, color='black', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.show()

# 8. Analyze feature correlations
print("\n" + "="*80)
print("FEATURE CORRELATION ANALYSIS:")
print("="*80)

# Get top 3 features with highest absolute loadings on PC1
top_features_pc1 = loadings_df['PC1'].abs().sort_values(ascending=False).head(3).index
print(f"\nTop 3 features with highest loadings on PC1: {list(top_features_pc1)}")

# Show correlation between these features
correlation_matrix = df[list(top_features_pc1)].corr()
print("\nCorrelation matrix of top features:")
print(correlation_matrix.round(3))
