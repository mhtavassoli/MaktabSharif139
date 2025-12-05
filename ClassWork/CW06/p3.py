# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

# Load the dataset
data = load_breast_cancer()
X = data.data[:, :8]  # Use first 8 features
feature_names = data.feature_names[:8]

# Create DataFrame
df = pd.DataFrame(X, columns=feature_names)

# Calculate correlation matrix
correlation_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix Heatmap - Breast Cancer Dataset (First 8 Features)')
plt.tight_layout()
plt.show()

# Find top positive and negative correlations
correlation_pairs = []

# Get all unique pairs of features
for i in range(len(feature_names)):
    for j in range(i+1, len(feature_names)):
        corr_value = correlation_matrix.iloc[i, j]
        correlation_pairs.append({
            'feature1': feature_names[i],
            'feature2': feature_names[j],
            'correlation': corr_value
        })

# Convert to DataFrame for easier manipulation
corr_df = pd.DataFrame(correlation_pairs)

# Find top 3 positive correlations
top_positive = corr_df.nlargest(3, 'correlation')

# Find top 3 negative correlations
top_negative = corr_df.nsmallest(3, 'correlation')

print("Top 3 Positive Correlations:")
print(top_positive.to_string(index=False))
print("\nTop 3 Negative Correlations:")
print(top_negative.to_string(index=False))
