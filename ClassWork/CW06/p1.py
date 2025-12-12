# Import required libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import seaborn as sns

# Load the diabetes dataset
diabetes = load_diabetes()
print("Feature names:", diabetes.feature_names)

# Create DataFrame with only the 10 numeric features
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
print("DataFrame shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Calculate 10x10 covariance matrix
cov_matrix = df.cov()
print("\n" + "="*50)
print("10x10 COVARIANCE MATRIX:")
print("="*50)
print(cov_matrix)

# Find top 3 pairs with highest positive covariance
print("\n" + "="*50)
print("TOP 3 POSITIVE COVARIANCE PAIRS:")
print("="*50)

# Flatten the covariance matrix and exclude diagonal (variance) elements
cov_pairs = []
for i in range(len(cov_matrix.columns)):
    for j in range(i+1, len(cov_matrix.columns)):  # Avoid duplicates and diagonal
        feature1 = cov_matrix.columns[i]
        feature2 = cov_matrix.columns[j]
        covariance = cov_matrix.iloc[i, j]
        cov_pairs.append((feature1, feature2, covariance))

# Sort by absolute covariance value
cov_pairs_sorted = sorted(cov_pairs, key=lambda x: x[2], reverse=True)

# Display top 3 positive
for i, (f1, f2, cov) in enumerate(cov_pairs_sorted[:3]):
    print(f"{i+1}. {f1} - {f2}: {cov:.6f}")

# Display top 3 negative (lowest values)
print("\nTOP 3 NEGATIVE COVARIANCE PAIRS:")
print("="*50)
cov_pairs_neg_sorted = sorted(cov_pairs, key=lambda x: x[2])  # Sort ascending
for i, (f1, f2, cov) in enumerate(cov_pairs_neg_sorted[:3]):
    print(f"{i+1}. {f1} - {f2}: {cov:.6f}")

# Calculate correlation matrix
corr_matrix = df.corr()
print("\n" + "="*50)
print("CORRELATION MATRIX:")
print("="*50)
print(corr_matrix)

# Visualization
plt.figure(figsize=(15, 6))

# Heatmap for covariance matrix
plt.subplot(1, 2, 1)
sns.heatmap(cov_matrix, annot=True, fmt=".3f", cmap='coolwarm', center=0)
plt.title('Covariance Matrix Heatmap')

# Heatmap for correlation matrix
plt.subplot(1, 2, 2)
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap='coolwarm', center=0)
plt.title('Correlation Matrix Heatmap')

plt.tight_layout()
plt.show()

# Additional analysis: Compare specific pairs with high covariance but potentially different correlation
print("\n" + "="*50)
print("DETAILED COMPARISON OF SELECTED PAIRS:")
print("="*50)

# Let's examine the top covariance pairs in detail
for f1, f2, cov in cov_pairs_sorted[:3]:
    corr = corr_matrix.loc[f1, f2]
    print(f"\nPair: {f1} - {f2}")
    print(f"Covariance: {cov:.6f}")
    print(f"Correlation: {corr:.6f}")
    print(f"Std {f1}: {df[f1].std():.6f}")
    print(f"Std {f2}: {df[f2].std():.6f}")
    