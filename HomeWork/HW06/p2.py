# Import required libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from scipy.spatial.distance import euclidean, cityblock, chebyshev, cosine
from scipy.linalg import inv
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = load_breast_cancer()

# Create a DataFrame with features
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add target column
df['target'] = data.target
df['target_name'] = data.target_names[data.target]

# Display dataset shape
print("Dataset Shape:", df.shape)
print("Number of Features:", len(data.feature_names))
print("Number of Samples:", len(df))
print("\nClass Distribution:")
print(df['target_name'].value_counts())

# Display first few rows
print("\nFirst 5 rows of the dataset:")
print(df.iloc[:5, :5].join(df[['target', 'target_name']].iloc[:5]))

############################################################################# part 1

# Select reference samples (3 from each class)
benign_samples = df[df['target'] == 0].iloc[:3]
malignant_samples = df[df['target'] == 1].iloc[:3]

# Combine reference samples
reference_samples = pd.concat([benign_samples, malignant_samples])
reference_indices = reference_samples.index.tolist()

# Extract features only (without target columns)
X = df.drop(['target', 'target_name'], axis=1)
X_ref = reference_samples.drop(['target', 'target_name'], axis=1)

# Calculate covariance matrix for Mahalanobis distance
cov_matrix = np.cov(X.values.T)
try:
    inv_cov_matrix = inv(cov_matrix)
except:
    # Use pseudo-inverse if matrix is singular
    inv_cov_matrix = np.linalg.pinv(cov_matrix)

# Function to calculate Mahalanobis distance
def mahalanobis_distance(x, y, inv_cov):
    diff = x - y
    return np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))

# Prepare results table
results = []

# Calculate distances for all pairs of reference samples
for i in range(len(reference_indices)):
    for j in range(i+1, len(reference_indices)):
        sample1_idx = reference_indices[i]
        sample2_idx = reference_indices[j]
        
        # Get sample data
        x1 = X_ref.loc[sample1_idx].values
        x2 = X_ref.loc[sample2_idx].values
        
        # Get class labels
        class1 = reference_samples.loc[sample1_idx, 'target_name']
        class2 = reference_samples.loc[sample2_idx, 'target_name']
        
        # Calculate distances
        euclidean_dist = euclidean(x1, x2)
        manhattan_dist = cityblock(x1, x2)
        chebyshev_dist = chebyshev(x1, x2)
        cosine_dist = cosine(x1, x2)
        mahalanobis_dist = mahalanobis_distance(x1, x2, inv_cov_matrix)
        
        # Store results
        results.append({
            'Sample1': sample1_idx,
            'Sample2': sample2_idx,
            'Class1': class1,
            'Class2': class2,
            'Euclidean': round(euclidean_dist, 4),
            'Manhattan': round(manhattan_dist, 4),
            'Chebyshev': round(chebyshev_dist, 4),
            'Cosine': round(cosine_dist, 4),
            'Mahalanobis': round(mahalanobis_dist, 4)
        })

# Create results DataFrame
results_df = pd.DataFrame(results)
print("\nDistance Comparison Between Reference Samples:")
print(results_df.to_string(index=False)) 

############################################################################# part 2

# Scenario implementation for an early warning system
def early_warning_system(distance_metric='mahalanobis', threshold_percentile=95):
    """
    Design an early warning system based on distance metrics
    """
    # Separate malignant and benign samples
    malignant_data = X[df['target'] == 1].values
    benign_data = X[df['target'] == 0].values
    
    # Calculate centroids
    malignant_centroid = np.mean(malignant_data, axis=0)
    benign_centroid = np.mean(benign_data, axis=0)
    
    # Calculate distances for all samples to malignant centroid
    distances = []
    for idx, sample in X.iterrows():
        sample_values = sample.values
        
        if distance_metric == 'euclidean':
            dist = euclidean(sample_values, malignant_centroid)
        elif distance_metric == 'manhattan':
            dist = cityblock(sample_values, malignant_centroid)
        elif distance_metric == 'chebyshev':
            dist = chebyshev(sample_values, malignant_centroid)
        elif distance_metric == 'cosine':
            dist = cosine(sample_values, malignant_centroid)
        elif distance_metric == 'mahalanobis':
            dist = mahalanobis_distance(sample_values, malignant_centroid, inv_cov_matrix)
        else:
            raise ValueError("Invalid distance metric")
        
        distances.append(dist)
    
    # Determine threshold
    threshold = np.percentile(distances, threshold_percentile)
    
    # Flag suspicious samples
    suspicious_indices = []
    for idx, dist in enumerate(distances):
        if dist < threshold:  # Closer to malignant centroid
            suspicious_indices.append(idx)
    
    return suspicious_indices, distances, threshold

# Test the warning system with different distance metrics
print("\nEarly Warning System Analysis:")
print("="*50)

for metric in ['euclidean', 'manhattan', 'cosine', 'mahalanobis']:
    suspicious, distances, threshold = early_warning_system(distance_metric=metric)
    actual_malignant = sum(df.iloc[suspicious]['target'] == 1)
    false_positives = sum(df.iloc[suspicious]['target'] == 0)
    
    print(f"\nUsing {metric.capitalize()} Distance:")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Flagged {len(suspicious)} suspicious samples")
    print(f"  Correctly identified {actual_malignant} malignant samples")
    print(f"  False positives: {false_positives}")
    print(f"  Accuracy: {actual_malignant/len(suspicious)*100:.1f}% among flagged samples")
    
    ############################################################################# part 3

