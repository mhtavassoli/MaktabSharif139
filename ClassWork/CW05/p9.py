import pandas as pd
import numpy as np

# Step 1: Read the data
# Read the CSV file
df=pd.read_csv(r"E:\Maktab\Artificial Intelligence\Programming\ClassWork\CW05\p9\iris_dataset.csv") 

# Check data types and basic info
print("Data Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# Rename the column
df = df.rename(columns={'sepal length (cm)': 'sepal_length'})

print("Column names in dataset:")
print(df.columns.tolist())

# 1- Select Sepal Length feature from Iris dataset as X
X = df['sepal_length']

print("=== Original Dataset (X) ===")
print(f"First 5 values of X: {X.head().tolist()}")

# 2- Calculate statistics for X
mean_X = np.mean(X)
variance_X = np.var(X)
range_X = np.max(X) - np.min(X)

print("\n=== Statistics for X (Original) ===")
print(f"Mean of X: {mean_X:.4f}")
print(f"Variance of X: {variance_X:.4f}")
print(f"Range of X: {range_X:.4f}")

# 3- Create new variable Y = 2X + 5
Y = 2 * X + 5

print("\n=== Transformed Dataset (Y) ===")
print(f"First 5 values of Y: {Y.head().tolist()}")

# 4- Calculate statistics for Y
mean_Y = np.mean(Y)
variance_Y = np.var(Y)
range_Y = np.max(Y) - np.min(Y)

print("\n=== Statistics for Y (Transformed) ===")
print(f"Mean of Y: {mean_Y:.4f}")
print(f"Variance of Y: {variance_Y:.4f}")
print(f"Range of Y: {range_Y:.4f}")

# 5- Compare and report changes
print("\n=== Comparison and Change Analysis ===")
print(f"Mean change: Y = 2 * {mean_X:.4f} + 5 = {mean_Y:.4f}")
print(f"Variance change: Variance_Y = 4 * Variance_X = 4 * {variance_X:.4f} = {variance_Y:.4f}")
print(f"Range change: Range_Y = 2 * Range_X = 2 * {range_X:.4f} = {range_Y:.4f}")

# Verification of transformation effects
print("\n=== Verification of Transformation Rules ===")
print(f"Expected mean of Y: {2 * mean_X + 5:.4f}")
print(f"Actual mean of Y: {mean_Y:.4f}")
print(f"Expected variance of Y: {4 * variance_X:.4f}")
print(f"Actual variance of Y: {variance_Y:.4f}")
