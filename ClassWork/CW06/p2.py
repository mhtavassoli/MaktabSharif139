# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

# Load the wine dataset
wine = load_wine()
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Extract the two columns: alcohol and magnesium
alcohol = wine_df['alcohol']
magnesium = wine_df['magnesium']

print("=== Original Data ===")
print(f"Alcohol sample: {alcohol[:5].values}")
print(f"Magnesium sample: {magnesium[:5].values}")

# Calculate original covariance and correlation
original_cov = np.cov(alcohol, magnesium)[0, 1]
original_corr = np.corrcoef(alcohol, magnesium)[0, 1]

print(f"\nOriginal Covariance: {original_cov:.4f}")
print(f"Original Pearson Correlation: {original_corr:.4f}")

# Multiply magnesium column by 20
magnesium_scaled = magnesium * 20

print(f"\n=== After Scaling Magnesium by 20 ===")
print(f"Magnesium sample (scaled): {magnesium_scaled[:5].values}")

# Calculate covariance and correlation after scaling
scaled_cov = np.cov(alcohol, magnesium_scaled)[0, 1]
scaled_corr = np.corrcoef(alcohol, magnesium_scaled)[0, 1]

print(f"Scaled Covariance: {scaled_cov:.4f}")
print(f"Scaled Pearson Correlation: {scaled_corr:.4f}")

# Verification
print(f"\n=== Verification ===")
print(f"Covariance ratio (scaled/original): {scaled_cov/original_cov:.2f}")
print(f"Correlation difference: {abs(scaled_corr - original_corr):.10f}")
