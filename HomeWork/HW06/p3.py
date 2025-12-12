# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
feature_names = diabetes.feature_names
target = diabetes.target

print("Dataset shape:", X.shape)
print("Feature names:", feature_names)
print("\nOriginal data - First 5 rows:")
print(pd.DataFrame(X[:5], columns=feature_names))

# Standardize the data (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a DataFrame for the scaled data
df_scaled = pd.DataFrame(X_scaled, columns=feature_names)

# Calculate and display summary statistics
summary_stats = pd.DataFrame({
    'Mean': df_scaled.mean(),
    'Std': df_scaled.std()
})

print("\n" + "="*60)
print("SCALED DATA SUMMARY")
print("="*60)
print(summary_stats)
print("\nVerification - Mean of scaled data (should be near 0):")
print(f"Overall mean: {X_scaled.mean():.6f}")
print("\nVerification - Std of scaled data (should be 1):")
print(f"Overall std: {X_scaled.std():.6f}")
# ---------------------------------------------------------------------- part 1
# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Calculate explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Create a summary table
pca_summary = pd.DataFrame({
    'Component': range(1, len(explained_variance) + 1),
    'Explained Variance': explained_variance,
    'Cumulative Variance': cumulative_variance
})

print("\n" + "="*60)
print("PCA ANALYSIS - EXPLAINED VARIANCE")
print("="*60)
print(pca_summary.to_string(index=False))

# Determine components needed for different variance thresholds
thresholds = [0.80, 0.90, 0.95, 0.99]
print("\n" + "="*60)
print("COMPONENTS NEEDED FOR DIFFERENT VARIANCE THRESHOLDS")
print("="*60)
for threshold in thresholds:
    n_components = np.argmax(cumulative_variance >= threshold) + 1
    print(f"For {threshold*100:.0f}% variance: {n_components} components needed "
          f"(cumulative variance: {cumulative_variance[n_components-1]:.3f})")

# Visualization 1: Scree plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, color='skyblue')
plt.plot(range(1, len(explained_variance) + 1), explained_variance, 'ro-', linewidth=2)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.grid(True, alpha=0.3)

# Visualization 2: Cumulative variance plot
plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-', linewidth=2)
plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80% threshold')
plt.axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='90% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# ------------------------------------------------------------------------------part 2 
# Analyze loadings (components) for first few principal components
print("\n" + "="*60)
print("COMPONENT LOADINGS ANALYSIS")
print("="*60)

# Get the components (loadings)
components = pca.components_

# Analyze first 3 principal components
for i in range(3):
    print(f"\nPrincipal Component {i+1}:")
    print("-" * 40)
    
    # Create a DataFrame for better visualization
    loadings_df = pd.DataFrame({
        'Feature': feature_names,
        'Loading': components[i]
    })
    
    # Sort by absolute value for importance
    loadings_df['Absolute Loading'] = np.abs(loadings_df['Loading'])
    loadings_df = loadings_df.sort_values('Absolute Loading', ascending=False)
    
    print(loadings_df[['Feature', 'Loading']].to_string(index=False))
    
    # Interpretation
    print(f"\nInterpretation of PC{i+1}:")
    top_features = loadings_df.head(3)
    print(f"Top features: {', '.join(top_features['Feature'].values)}")
    
    # Check if loadings are mostly positive or negative
    pos_features = loadings_df[loadings_df['Loading'] > 0.2]['Feature'].tolist()
    neg_features = loadings_df[loadings_df['Loading'] < -0.2]['Feature'].tolist()
    
    if pos_features:
        print(f"Strong positive correlation with: {', '.join(pos_features)}")
    if neg_features:
        print(f"Strong negative correlation with: {', '.join(neg_features)}")

# Visualize loadings for first two components
plt.figure(figsize=(10, 6))
for i, feature in enumerate(feature_names):
    plt.arrow(0, 0, components[0, i], components[1, i], 
              head_width=0.03, head_length=0.03, fc='blue', ec='blue', alpha=0.7)
    plt.text(components[0, i]*1.15, components[1, i]*1.15, 
             feature, color='darkblue', ha='center', va='center')

plt.xlabel(f'Principal Component 1 ({explained_variance[0]*100:.1f}% variance)')
plt.ylabel(f'Principal Component 2 ({explained_variance[1]*100:.1f}% variance)')
plt.title('Feature Loadings on First Two Principal Components')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()
# ----------------------------------------------------------------------------------- part 3
# Create 2D visualization using first two principal components
plt.figure(figsize=(12, 8))

# Color by target value (quantiles)
target_quantiles = pd.qcut(target, q=4, labels=['Very Low', 'Low', 'High', 'Very High'])
colors = {'Very Low': 'green', 'Low': 'lightgreen', 'High': 'orange', 'Very High': 'red'}
color_list = [colors[q] for q in target_quantiles]

# Scatter plot
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                      c=color_list, alpha=0.7, s=50, edgecolors='k', linewidth=0.5)

# Create legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=colors[q], markeredgecolor='k', 
                          markersize=10, label=q) 
                   for q in ['Very Low', 'Low', 'High', 'Very High']]

plt.xlabel(f'Principal Component 1 ({explained_variance[0]*100:.1f}% variance)')
plt.ylabel(f'Principal Component 2 ({explained_variance[1]*100:.1f}% variance)')
plt.title('2D PCA Projection of Diabetes Dataset\n(Colored by Disease Progression Quantiles)')
plt.legend(handles=legend_elements, title='Disease Progression')
plt.grid(True, alpha=0.3)

# Add density contours
from scipy.stats import gaussian_kde
xy = np.vstack([X_pca[:, 0], X_pca[:, 1]])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
plt.scatter(X_pca[idx, 0], X_pca[idx, 1], c=z[idx], s=50, alpha=0.1, cmap='viridis')

plt.tight_layout()
plt.show()

# Qualitative analysis of patterns
print("\n" + "="*60)
print("QUALITATIVE ANALYSIS OF 2D VISUALIZATION")
print("="*60)
print("\nPatterns observed in the 2D PCA projection:")
print("1. Data Distribution: The points are spread across the PC1-PC2 plane")
print("2. Density Patterns: Higher density areas suggest clusters of similar patients")
print("3. Target Relationship: Colors show some grouping by disease progression")
print("4. Outliers: Some points appear far from the main concentration")
print("5. Linear Patterns: Potential linear relationships along PC1 axis")

# ------------------------------------------------------------------------------------ part 4
