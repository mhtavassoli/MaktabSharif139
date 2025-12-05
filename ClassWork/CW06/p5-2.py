import numpy as np
from sklearn.datasets import load_iris

# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

print("Dataset loaded successfully!")
print(f"Features: {iris.feature_names}")
print(f"Classes: {target_names}")

# Custom Distance Functions

def euclidean_distance(x, y):
    """
    Calculate Euclidean distance between two vectors
    
    Formula: sqrt(Σ(x_i - y_i)²)
    Measures straight-line distance in n-dimensional space
    """
    return np.sqrt(np.sum((x - y) ** 2))

def manhattan_distance(x, y):
    """
    Calculate Manhattan distance (L1 norm) between two vectors
    
    Formula: Σ|x_i - y_i|
    Measures distance along axes at right angles
    """
    return np.sum(np.abs(x - y))

def chebyshev_distance(x, y):
    """
    Calculate Chebyshev distance between two vectors
    
    Formula: max(|x_i - y_i|)
    Measures maximum difference along any coordinate dimension
    """
    return np.max(np.abs(x - y))

def cosine_distance(x, y):
    """
    Calculate Cosine distance between two vectors
    
    Formula: 1 - (x·y) / (||x|| * ||y||)
    Measures angular difference regardless of magnitude
    """
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    similarity = dot_product / (norm_x * norm_y)
    return 1 - similarity

def mahalanobis_distance(x, y, cov_matrix):
    """
    Calculate Mahalanobis distance between two vectors
    
    Formula: sqrt((x-y)ᵀ * Σ⁻¹ * (x-y))
    Measures distance considering data covariance structure
    """
    diff = x - y
    try:
        inv_cov = np.linalg.inv(cov_matrix)
        return np.sqrt(diff.T @ inv_cov @ diff)
    except np.linalg.LinAlgError:
        return np.nan

# Calculate class means
print("\nCLASS MEANS:")
class_means = []
for i in range(3):
    class_data = X[y == i]
    class_mean = np.mean(class_data, axis=0)
    class_means.append(class_mean)
    print(f"{target_names[i]}: {class_mean}")

class_means = np.array(class_means)

# Calculate covariance matrix for Mahalanobis distance
cov_matrix = np.cov(X.T)

# Calculate all distances
print("\n" + "="*60)
print("DISTANCE COMPARISON BETWEEN CLASS MEANS")
print("="*60)

distance_results = {}

for i in range(3):
    for j in range(i+1, 3):
        print(f"\n--- {target_names[i]} vs {target_names[j]} ---")
        
        euclidean = euclidean_distance(class_means[i], class_means[j])
        manhattan = manhattan_distance(class_means[i], class_means[j])
        chebyshev = chebyshev_distance(class_means[i], class_means[j])
        cosine = cosine_distance(class_means[i], class_means[j])
        mahalanobis_dist = mahalanobis_distance(class_means[i], class_means[j], cov_matrix)
        
        print(f"Euclidean:   {euclidean:.4f}")
        print(f"Manhattan:   {manhattan:.4f}")
        print(f"Chebyshev:   {chebyshev:.4f}")
        print(f"Cosine:      {cosine:.4f}")
        print(f"Mahalanobis: {mahalanobis_dist:.4f}")
        
        # Store results for analysis
        pair_name = f"{target_names[i]}-{target_names[j]}"
        distance_results[pair_name] = {
            'Euclidean': euclidean,
            'Manhattan': manhattan,
            'Chebyshev': chebyshev,
            'Cosine': cosine,
            'Mahalanobis': mahalanobis_dist
        }

# Compare separation power
print("\n" + "="*50)
print("SEPARATION ANALYSIS")
print("="*50)

def calculate_separation_score(distances_dict):
    """Calculate average separation for each distance metric"""
    separation_scores = {}
    metrics = ['Euclidean', 'Manhattan', 'Chebyshev', 'Cosine', 'Mahalanobis']
    
    for metric in metrics:
        total_dist = 0
        count = 0
        for pair in distances_dict:
            if not np.isnan(distances_dict[pair][metric]):
                total_dist += distances_dict[pair][metric]
                count += 1
        separation_scores[metric] = total_dist / count if count > 0 else 0
    
    return separation_scores

separation_scores = calculate_separation_score(distance_results)

print("Average separation between classes:")
for metric, score in separation_scores.items():
    print(f"{metric:12}: {score:.4f}")

best_metric = max(separation_scores, key=separation_scores.get)
print(f"\nBest separating distance: {best_metric}")
print(f"Average separation: {separation_scores[best_metric]:.4f}")
