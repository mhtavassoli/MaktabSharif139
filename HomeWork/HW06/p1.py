# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
import ssl

# Disable SSL verification (not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context

# Load the California Housing dataset
california = fetch_california_housing()

# Convert to DataFrame
df = pd.DataFrame(california.data, columns=california.feature_names)
df['MedHouseVal'] = california.target

# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData Types:")
print(df.dtypes)
print("\nBasic Statistics:")
print(df.describe())

# Display range of values for each column
print("\nRange of values for each column:")
for col in df.columns:
    min_val = df[col].min()
    max_val = df[col].max()
    print(f"{col}: [{min_val:.4f}, {max_val:.4f}]")

print(f"\nTotal rows: {df.shape[0]}")
print(f"Total columns: {df.shape[1]}")
########################################################################## part 1
# Calculate covariance matrix
cov_matrix = df.cov()
print("Covariance Matrix:")
print(cov_matrix)

# Calculate correlation matrix
corr_matrix = df.corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Heatmap for covariance matrix
sns.heatmap(cov_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, cbar_kws={"shrink": 0.8}, ax=axes[0])
axes[0].set_title('Covariance Matrix Heatmap')
axes[0].tick_params(axis='x', rotation=45)
axes[0].tick_params(axis='y', rotation=0)

# Heatmap for correlation matrix
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, cbar_kws={"shrink": 0.8}, ax=axes[1])
axes[1].set_title('Correlation Matrix Heatmap')
axes[1].tick_params(axis='x', rotation=45)
axes[1].tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.show()

# Numerical comparison
print("\nComparison of Covariance and Correlation for MedHouseVal:")
for col in df.columns:
    if col != 'MedHouseVal':
        cov_val = cov_matrix.loc['MedHouseVal', col]
        corr_val = corr_matrix.loc['MedHouseVal', col]
        print(f"{col}: Covariance = {cov_val:.4f}, Correlation = {corr_val:.4f}")
########################################################################## part 2
# Find features with highest correlation with house price
target_corr = corr_matrix['MedHouseVal'].sort_values(ascending=False)
print("Correlation with MedHouseVal (sorted):")
print(target_corr)

# Visualization of correlations with target
plt.figure(figsize=(10, 6))
bars = plt.barh(target_corr.index[:-1], target_corr.values[:-1], 
                color=['green' if x > 0 else 'red' for x in target_corr.values[:-1]])
plt.xlabel('Correlation Coefficient')
plt.title('Correlation of Features with Median House Value')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.grid(axis='x', alpha=0.3)

# Add value labels
for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', ha='left' if width > 0 else 'right', 
             va='center', fontweight='bold')

plt.tight_layout()
plt.show()
########################################################################## part 3
# Find pairs where covariance and correlation tell different stories
comparison_data = []

for i in range(len(df.columns)):
    for j in range(i+1, len(df.columns)):
        col1 = df.columns[i]
        col2 = df.columns[j]
        cov = cov_matrix.loc[col1, col2]
        corr = corr_matrix.loc[col1, col2]
        
        # Calculate absolute values
        abs_cov = abs(cov)
        abs_corr = abs(corr)
        
        # Check for interesting cases
        if abs_cov > 1000 and abs_corr < 0.5:  # High covariance, low correlation
            comparison_data.append({
                'Pair': f'{col1} - {col2}',
                'Covariance': cov,
                'Correlation': corr,
                'Type': 'High Cov, Low Corr'
            })
        elif abs_cov < 100 and abs_corr > 0.7:  # Low covariance, high correlation
            comparison_data.append({
                'Pair': f'{col1} - {col2}',
                'Covariance': cov,
                'Correlation': corr,
                'Type': 'Low Cov, High Corr'
            })

# Display interesting cases
if comparison_data:
    comparison_df = pd.DataFrame(comparison_data)
    print("Interesting cases where covariance and correlation differ significantly:")
    print(comparison_df)

# Example analysis for specific pairs
print("\nSpecific example analysis:")
print(f"Population - AveOccup:")
print(f"  Covariance: {cov_matrix.loc['Population', 'AveOccup']:.2f}")
print(f"  Correlation: {corr_matrix.loc['Population', 'AveOccup']:.2f}")
print("\nMedInc - MedHouseVal:")
print(f"  Covariance: {cov_matrix.loc['MedInc', 'MedHouseVal']:.4f}")
print(f"  Correlation: {corr_matrix.loc['MedInc', 'MedHouseVal']:.4f}")
########################################################################## part 4
# Demonstrate effect of changing units
example_feature = 'AveRooms'

# Original values
original_cov = cov_matrix.loc[example_feature, 'MedHouseVal']
original_corr = corr_matrix.loc[example_feature, 'MedHouseVal']

# Create scaled version (multiply by 1000)
scaled_feature = df[example_feature] * 1000

# Calculate covariance and correlation with scaled feature
scaled_cov = np.cov(scaled_feature, df['MedHouseVal'])[0, 1]
scaled_corr = np.corrcoef(scaled_feature, df['MedHouseVal'])[0, 1]

print("Effect of changing units (multiplying by 1000):")
print(f"Original - {example_feature} vs MedHouseVal:")
print(f"  Covariance: {original_cov:.6f}")
print(f"  Correlation: {original_corr:.6f}")
print(f"\nScaled - {example_feature}×1000 vs MedHouseVal:")
print(f"  Covariance: {scaled_cov:.6f} (changed by factor of {scaled_cov/original_cov:.0f})")
print(f"  Correlation: {scaled_corr:.6f} (unchanged)")
########################################################################## part 5
