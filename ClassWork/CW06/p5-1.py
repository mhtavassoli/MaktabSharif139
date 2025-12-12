import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_similarity
from scipy.spatial.distance import chebyshev, mahalanobis

# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

print("Dataset loaded successfully!")
print(f"Features: {iris.feature_names}")
print(f"Classes: {target_names}")

# Calculate mean for each class
class_means = []
for i in range(3):  # 3 classes: setosa, versicolor, virginica
    class_data = X[y == i]
    class_mean = np.mean(class_data, axis=0)
    class_means.append(class_mean)
    print(f"\n{target_names[i]} mean: {class_mean}")

class_means = np.array(class_means)

# Calculate distances between class means
print("\n" + "="*50)
print("DISTANCES BETWEEN CLASS MEANS")
print("="*50)

# Euclidean Distance
print("\nEUCLIDEAN DISTANCE:")
for i in range(3):
    for j in range(i+1, 3):
        dist = euclidean_distances([class_means[i]], [class_means[j]])[0][0]
        print(f"{target_names[i]} - {target_names[j]}: {dist:.4f}")

# Manhattan Distance
print("\nMANHATTAN DISTANCE:")
for i in range(3):
    for j in range(i+1, 3):
        dist = manhattan_distances([class_means[i]], [class_means[j]])[0][0]
        print(f"{target_names[i]} - {target_names[j]}: {dist:.4f}")

# Chebyshev Distance
print("\nCHEBYSHEV DISTANCE:")
for i in range(3):
    for j in range(i+1, 3):
        dist = chebyshev(class_means[i], class_means[j])
        print(f"{target_names[i]} - {target_names[j]}: {dist:.4f}")

# Cosine Distance (1 - cosine similarity)
print("\nCOSINE DISTANCE:")
for i in range(3):
    for j in range(i+1, 3):
        similarity = cosine_similarity([class_means[i]], [class_means[j]])[0][0]
        dist = 1 - similarity
        print(f"{target_names[i]} - {target_names[j]}: {dist:.4f}")

# Mahalanobis Distance (using dataset covariance)
print("\nMAHALANOBIS DISTANCE:")
# Calculate covariance matrix of entire dataset
cov_matrix = np.cov(X.T)
# Calculate inverse covariance matrix for Mahalanobis
try:
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    for i in range(3):
        for j in range(i+1, 3):
            dist = mahalanobis(class_means[i], class_means[j], inv_cov_matrix)
            print(f"{target_names[i]} - {target_names[j]}: {dist:.4f}")
except np.linalg.LinAlgError:
    print("Cannot compute Mahalanobis distance - covariance matrix is singular")

# Additional analysis: Compare separation power
print("\n" + "="*50)
print("SEPARATION ANALYSIS")
print("="*50)

# Calculate average distance between classes for each metric
def calculate_avg_separation(distances_dict):
    total = sum(distances_dict.values())
    return total / len(distances_dict)

# Collect all distances
euclidean_dists = {}
manhattan_dists = {}
chebyshev_dists = {}
cosine_dists = {}
mahalanobis_dists = {}

for i in range(3):
    for j in range(i+1, 3):
        pair_name = f"{target_names[i]}-{target_names[j]}"
        
        euclidean_dists[pair_name] = euclidean_distances([class_means[i]], [class_means[j]])[0][0]
        manhattan_dists[pair_name] = manhattan_distances([class_means[i]], [class_means[j]])[0][0]
        chebyshev_dists[pair_name] = chebyshev(class_means[i], class_means[j])
        
        similarity = cosine_similarity([class_means[i]], [class_means[j]])[0][0]
        cosine_dists[pair_name] = 1 - similarity
        
        if 'inv_cov_matrix' in locals():
            mahalanobis_dists[pair_name] = mahalanobis(class_means[i], class_means[j], inv_cov_matrix)

print("Average separation between classes:")
print(f"Euclidean: {calculate_avg_separation(euclidean_dists):.4f}")
print(f"Manhattan: {calculate_avg_separation(manhattan_dists):.4f}")
print(f"Chebyshev: {calculate_avg_separation(chebyshev_dists):.4f}")
print(f"Cosine: {calculate_avg_separation(cosine_dists):.4f}")
if mahalanobis_dists:
    print(f"Mahalanobis: {calculate_avg_separation(mahalanobis_dists):.4f}")

# Find which distance provides the best separation
if mahalanobis_dists:
    separations = {
        'Euclidean': calculate_avg_separation(euclidean_dists),
        'Manhattan': calculate_avg_separation(manhattan_dists),
        'Chebyshev': calculate_avg_separation(chebyshev_dists),
        'Cosine': calculate_avg_separation(cosine_dists),
        'Mahalanobis': calculate_avg_separation(mahalanobis_dists)
    }
else:
    separations = {
        'Euclidean': calculate_avg_separation(euclidean_dists),
        'Manhattan': calculate_avg_separation(manhattan_dists),
        'Chebyshev': calculate_avg_separation(chebyshev_dists),
        'Cosine': calculate_avg_separation(cosine_dists)
    }

best_distance = max(separations, key=separations.get)
print(f"\nBest separating distance: {best_distance} with average separation: {separations[best_distance]:.4f}")
