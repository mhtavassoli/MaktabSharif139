# -*- coding: utf-8 -*-
"""
Wine Classification Project: KNN, PCA, Rule-Based, and Regression Analysis
Authors: ML Project
Description: Comprehensive analysis of wine dataset using multiple ML techniques
"""

# ================ Import Libraries ================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import mahalanobis
import warnings
warnings.filterwarnings('ignore')

# ================ Load and Prepare Data ================
# Load wine dataset from sklearn
wine_data = load_wine()
X = wine_data.data  # Features (13 dimensions)
y = wine_data.target  # Target classes (0, 1, 2)
feature_names = wine_data.feature_names
target_names = wine_data.target_names

print("="*60)
print("WINE DATASET INFORMATION")
print("="*60)
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Feature names: {feature_names}")
print(f"Target names: {target_names}")
print(f"Class distribution: {np.bincount(y)}")

# Split data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Standardize features for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# ================ Stage 1: Simple KNN ================
print("\n" + "="*60)
print("STAGE 1: SIMPLE KNN CLASSIFICATION")
print("="*60)

# Create KNN model with k=3
knn_model = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn_model.fit(X_train_scaled, y_train)

# Make predictions and calculate accuracy
y_pred_knn = knn_model.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print(f"KNN (k=3, Euclidean) Accuracy: {accuracy_knn:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn, target_names=target_names))

# ================ Stage 2: Dimensionality Reduction with PCA ================
print("\n" + "="*60)
print("STAGE 2: DIMENSIONALITY REDUCTION WITH PCA")
print("="*60)

# Apply PCA to reduce from 13 to 2 principal components
pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Explained variance ratio by PC1 and PC2: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")

# Train KNN on PCA-reduced data
knn_pca = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn_pca.fit(X_train_pca, y_train)

# Make predictions and calculate accuracy
y_pred_knn_pca = knn_pca.predict(X_test_pca)
accuracy_knn_pca = accuracy_score(y_test, y_pred_knn_pca)

print(f"\nKNN on PCA-reduced data (k=3) Accuracy: {accuracy_knn_pca:.4f}")

# Visualize PCA results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, 
                      cmap='viridis', alpha=0.7, edgecolors='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Wine Dataset (Training Set)')
plt.colorbar(scatter, label='Wine Class')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, 
                      cmap='viridis', alpha=0.7, edgecolors='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Wine Dataset (Testing Set)')
plt.colorbar(scatter, label='Wine Class')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ================ Stage 3: Rule-Based System ================
print("\n" + "="*60)
print("STAGE 3: RULE-BASED CLASSIFICATION")
print("="*60)

# Find alcohol feature index
alcohol_idx = list(feature_names).index('alcohol')
print(f"Alcohol feature index: {alcohol_idx}")

# Extract alcohol values
alcohol_train = X_train[:, alcohol_idx]
alcohol_test = X_test[:, alcohol_idx]

# Rule-based classification function
def rule_based_classification(alcohol_values):
    """Classify wines based on alcohol content rules"""
    predictions = np.zeros_like(alcohol_values, dtype=int)
    
    # Apply rules
    predictions[alcohol_values < 12] = 0
    predictions[(alcohol_values >= 12) & (alcohol_values < 13)] = 1
    predictions[alcohol_values >= 13] = 2
    
    return predictions

# Make predictions using rule-based system
y_pred_rule = rule_based_classification(alcohol_test)
accuracy_rule = accuracy_score(y_test, y_pred_rule)

print(f"Rule-Based Classification Accuracy: {accuracy_rule:.4f}")
print("\nRule-Based Classification Report:")
print(classification_report(y_test, y_pred_rule, target_names=target_names))

# Compare with KNN
print("\nComparison of Accuracies:")
print(f"KNN (13 features): {accuracy_knn:.4f}")
print(f"KNN (PCA 2 features): {accuracy_knn_pca:.4f}")
print(f"Rule-Based (Alcohol only): {accuracy_rule:.4f}")

# ================ Stage 4: Effect of Distance Metrics ================
print("\n" + "="*60)
print("STAGE 4: EFFECT OF DISTANCE METRICS")
print("="*60)

# Define distance metrics to test
distance_metrics = ['euclidean', 'manhattan', 'chebyshev']

# Train KNN with different distance metrics on full data
accuracy_results = {}
for metric in distance_metrics:
    knn_metric = KNeighborsClassifier(n_neighbors=3, metric=metric)
    knn_metric.fit(X_train_scaled, y_train)
    y_pred_metric = knn_metric.predict(X_test_scaled)
    accuracy_results[metric] = accuracy_score(y_test, y_pred_metric)
    print(f"KNN with {metric.capitalize()} distance: {accuracy_results[metric]:.4f}")

# For cosine distance (requires different approach)
knn_cosine = KNeighborsClassifier(n_neighbors=3, metric='cosine')
knn_cosine.fit(X_train_scaled, y_train)
y_pred_cosine = knn_cosine.predict(X_test_scaled)
accuracy_results['cosine'] = accuracy_score(y_test, y_pred_cosine)
print(f"KNN with Cosine distance: {accuracy_results['cosine']:.4f}")

# Visualize decision boundaries for different metrics (using PCA data)
plt.figure(figsize=(15, 10))

# Create mesh grid for decision boundaries
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

for i, metric in enumerate(distance_metrics + ['cosine'], 1):
    # Train KNN on PCA data with specific metric
    knn_boundary = KNeighborsClassifier(n_neighbors=3, metric=metric)
    knn_boundary.fit(X_train_pca, y_train)
    
    # Predict on mesh grid
    Z = knn_boundary.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.subplot(2, 2, i)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, 
                         cmap='viridis', edgecolors='k', s=50)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'Decision Boundary: {metric.capitalize()} Distance')
    plt.colorbar(scatter, label='Wine Class')

plt.tight_layout()
plt.show()

# ================ Stage 5: Classification-as-Regression with Soft Encoding ================
print("\n" + "="*60)
print("STAGE 5: CLASSIFICATION-AS-REGRESSION WITH SOFT ENCODING")
print("="*60)

# Soft encoding: convert classes to continuous values
soft_encoding_map = {0: 0, 1: 0.5, 2: 1}
y_train_soft = np.array([soft_encoding_map[label] for label in y_train])
y_test_soft = np.array([soft_encoding_map[label] for label in y_test])

# Train linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train_soft)

# Make continuous predictions
y_pred_continuous = lin_reg.predict(X_test_scaled)

# Convert continuous predictions back to discrete classes
def continuous_to_class(continuous_values):
    """Convert continuous values to nearest class"""
    class_values = np.array([0, 0.5, 1])
    predictions = []
    
    for value in continuous_values:
        # Find nearest class value
        nearest_idx = np.argmin(np.abs(class_values - value))
        predictions.append([0, 1, 2][nearest_idx])
    
    return np.array(predictions)

y_pred_reg = continuous_to_class(y_pred_continuous)
accuracy_reg = accuracy_score(y_test, y_pred_reg)

print(f"Linear Regression with Soft Encoding Accuracy: {accuracy_reg:.4f}")
print("\nRegression-Based Classification Report:")
print(classification_report(y_test, y_pred_reg, target_names=target_names))

# Analyze bias in predictions
print("\nAnalysis of Soft Encoding Bias:")
print("Continuous predictions distribution:")
print(f"  Min: {y_pred_continuous.min():.4f}")
print(f"  Max: {y_pred_continuous.max():.4f}")
print(f"  Mean: {y_pred_continuous.mean():.4f}")
print(f"  Std: {y_pred_continuous.std():.4f}")

# Count predictions in each range
bins = [0, 0.25, 0.75, 1.1]
hist, _ = np.histogram(y_pred_continuous, bins=bins)
print("\nPrediction distribution in continuous space:")
print(f"  Near Class 0 (0-0.25): {hist[0]} predictions")
print(f"  Near Class 1 (0.25-0.75): {hist[1]} predictions")
print(f"  Near Class 2 (0.75-1.1): {hist[2]} predictions")

# Train regression on PCA data for visualization
lin_reg_pca = LinearRegression()
lin_reg_pca.fit(X_train_pca, y_train_soft)

# Create mesh grid
Z_continuous = lin_reg_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z_classes = continuous_to_class(Z_continuous)
Z_classes = Z_classes.reshape(xx.shape)

# Visualize decision boundary
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z_classes, alpha=0.3, cmap='viridis')
scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, 
                     cmap='viridis', edgecolors='k', s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Decision Boundary: Linear Regression with Soft Encoding')
plt.colorbar(scatter, label='Wine Class')
plt.grid(True, alpha=0.3)
plt.show()

# ================ Final Comparison ================
print("\n" + "="*60)
print("FINAL ACCURACY COMPARISON")
print("="*60)

methods = ['KNN (13D)', 'KNN (PCA 2D)', 'Rule-Based', 'Linear Regression']
accuracies = [accuracy_knn, accuracy_knn_pca, accuracy_rule, accuracy_reg]

for method, acc in zip(methods, accuracies):
    print(f"{method}: {acc:.4f}")

# Create comparison bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(methods, accuracies, color=['blue', 'orange', 'green', 'red'])
plt.ylim([0, 1.0])
plt.ylabel('Accuracy')
plt.title('Comparison of Classification Methods on Wine Dataset')
plt.grid(True, axis='y', alpha=0.3)

# Add accuracy values on top of bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# ================ Additional Analysis: Confusion Matrices ================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

models = [
    ('KNN (13D)', y_pred_knn),
    ('KNN (PCA)', y_pred_knn_pca),
    ('Rule-Based', y_pred_rule),
    ('Regression', y_pred_reg)
]

for idx, (title, y_pred) in enumerate(models):
    cm = confusion_matrix(y_test, y_pred)
    ax = axes[idx]
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f'Confusion Matrix: {title}')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)

plt.tight_layout()
plt.show()