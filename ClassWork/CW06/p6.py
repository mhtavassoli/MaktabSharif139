# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducible results
np.random.seed(42)

# Parameters
dim = 200  # Vector dimension
n_samples = 1000  # Number of samples for distribution display

# Generate vector X with normal distribution N(0,1)
X = np.random.normal(0, 1, dim)

# Create vectors according to the problem definition
A = 100 * X  # A = 100X
B = 250 * X  # B = 250X
C = np.random.normal(0, 1, dim)  # C = random noise vector

# Function to calculate Euclidean distance
def euclidean_distance(vec1, vec2):
    """
    Calculate Euclidean distance between two vectors.
    
    Formula: sqrt(∑(vec1_i - vec2_i)²)
    
    Parameters:
    vec1, vec2: numpy arrays of same dimension
    
    Returns:
    Euclidean distance (float)
    """
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

# Function to calculate cosine distance
def cosine_distance(vec1, vec2):
    """
    Calculate cosine distance and similarity between two vectors.
    
    Cosine similarity = (vec1·vec2) / (||vec1|| * ||vec2||)
    Cosine distance = 1 - cosine_similarity
    
    Parameters:
    vec1, vec2: numpy arrays of same dimension
    
    Returns:
    Tuple of (cosine_distance, cosine_similarity)
    """
    # Calculate dot product
    dot_product = np.dot(vec1, vec2)
    
    # Calculate norms
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Calculate cosine similarity and distance
    cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
    cosine_distance = 1 - cosine_similarity
    
    return cosine_distance, cosine_similarity

# Calculate distances between A and B
euclidean_AB = euclidean_distance(A, B)
cosine_dist_AB, cosine_sim_AB = cosine_distance(A, B)

# Calculate distances between A and C
euclidean_AC = euclidean_distance(A, C)
cosine_dist_AC, cosine_sim_AC = cosine_distance(A, C)

# Calculate distances between B and C
euclidean_BC = euclidean_distance(B, C)
cosine_dist_BC, cosine_sim_BC = cosine_distance(B, C)

# Display results
print("=" * 60)
print("ANALYSIS OF DISTANCES BETWEEN VECTORS")
print("=" * 60)
print(f"\nVector dimensions: {dim}")

# Display vector norms
print(f"\nVector A = 100 * X (Norm: {np.linalg.norm(A):.2f})")
print(f"Vector B = 250 * X (Norm: {np.linalg.norm(B):.2f})")
print(f"Vector C = Random noise (Norm: {np.linalg.norm(C):.2f})")

# Display distances between A and B
print("\n" + "=" * 60)
print("DISTANCES BETWEEN A AND B (PARALLEL VECTORS)")
print("=" * 60)
print(f"Euclidean distance: {euclidean_AB:.2f}")
print(f"Cosine similarity: {cosine_sim_AB:.6f}")
print(f"Cosine distance: {cosine_dist_AB:.6f}")

# Display distances between A and C
print("\n" + "=" * 60)
print("DISTANCES BETWEEN A AND C (RANDOM)")
print("=" * 60)
print(f"Euclidean distance: {euclidean_AC:.2f}")
print(f"Cosine similarity: {cosine_sim_AC:.6f}")
print(f"Cosine distance: {cosine_dist_AC:.6f}")

# Display distances between B and C
print("\n" + "=" * 60)
print("DISTANCES BETWEEN B AND C (RANDOM)")
print("=" * 60)
print(f"Euclidean distance: {euclidean_BC:.2f}")
print(f"Cosine similarity: {cosine_sim_BC:.6f}")
print(f"Cosine distance: {cosine_dist_BC:.6f}")

# Generate normal distribution samples for visualization
normal_samples = np.random.normal(0, 1, n_samples)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Histogram of normal distribution
axes[0].hist(normal_samples, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Mean (0)')
axes[0].set_title('Normal Distribution N(0,1) - Source of Vector X')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: 2D projection of vectors
# We'll project high-dimensional vectors to 2D for visualization
from sklearn.decomposition import PCA

# Combine vectors into a matrix
vectors = np.vstack([A, B, C])

# Apply PCA for dimensionality reduction to 2D
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

# Plot vectors in 2D space
colors = ['red', 'blue', 'green']
labels = ['A (100X)', 'B (250X)', 'C (Random)']

for i, (vec, color, label) in enumerate(zip(vectors_2d, colors, labels)):
    axes[1].scatter(vec[0], vec[1], color=color, s=150, label=label, alpha=0.8)
    axes[1].text(vec[0], vec[1], f'  {label}', fontsize=11, 
                 verticalalignment='center')

axes[1].set_title('2D Projection of Vectors (PCA)')
axes[1].set_xlabel('Principal Component 1')
axes[1].set_ylabel('Principal Component 2')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional mathematical analysis
print("\n" + "=" * 60)
print("MATHEMATICAL ANALYSIS")
print("=" * 60)

# Show why cosine similarity of A and B is 1
print("\nWhy cosine(A,B) = 1?")
print("A = 100X, B = 250X = 2.5 * A")
print("cosine_similarity = (A·B) / (||A|| * ||B||)")
print(f"A·B = {np.dot(A, B):.2f}")
print(f"||A|| = {np.linalg.norm(A):.2f}")
print(f"||B|| = {np.linalg.norm(B):.2f}")
print(f"||A|| * ||B|| = {np.linalg.norm(A) * np.linalg.norm(B):.2f}")
print(f"cosine_similarity = {np.dot(A, B)/(np.linalg.norm(A) * np.linalg.norm(B)):.6f}")

# Show Euclidean distance calculation
print("\n\nWhy Euclidean(A,B) is large?")
print(f"||A - B|| = sqrt(∑(A_i - B_i)²)")
print(f"For each dimension: B_i = 2.5 * A_i")
print(f"Difference per dimension: B_i - A_i = 150 * X_i")
print(f"Squared difference per dimension: (150 * X_i)²")
print(f"Sum over 200 dimensions = {np.sum((B - A)**2):.2f}")
print(f"Square root = {np.sqrt(np.sum((B - A)**2)):.2f}")