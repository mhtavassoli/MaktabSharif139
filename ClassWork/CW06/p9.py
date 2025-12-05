# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load the digits dataset
digits = load_digits()
X = digits.data  # Shape: (1797, 64) - 1797 images, each 8x8 flattened to 64 pixels
y = digits.target  # Labels for each image

# Display basic information about the dataset
print("Dataset shape:", X.shape)
print("Number of images:", X.shape[0])
print("Number of pixels per image:", X.shape[1])
print("Image dimensions: 8x8 pixels")

# Standardize the data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to the entire dataset
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

# Function to reconstruct images using k principal components
def reconstruct_image(pca_model, X_scaled, k, image_index):
    """
    Reconstruct a single image using k principal components
    
    Parameters:
    pca_model: Fitted PCA model
    X_scaled: Scaled original data
    k: Number of principal components to use
    image_index: Index of the image to reconstruct
    
    Returns:
    reconstructed_image: Reconstructed image using k components
    """
    # Get the PCA transformation with k components
    pca_k = PCA(n_components=k)
    X_pca_k = pca_k.fit_transform(X_scaled)
    
    # Reconstruct the image
    reconstructed = pca_k.inverse_transform(X_pca_k[image_index:image_index+1])
    
    return reconstructed[0]

# Select sample images for demonstration
sample_indices = [0, 100, 500]  # Indices of sample images to display
k_values = [5, 15, 30, 40]  # Different k values to test

# Create a figure to display original and reconstructed images
fig, axes = plt.subplots(len(sample_indices), len(k_values) + 1, figsize=(15, 8))
fig.suptitle('Image Reconstruction with Different Numbers of Principal Components', fontsize=14, y=1.02)

# Reshape function for visualization
def reshape_to_image(data_vector):
    """Reshape a 64-element vector to 8x8 image"""
    return data_vector.reshape(8, 8)

# Display original and reconstructed images
for row, idx in enumerate(sample_indices):
    # Display original image
    original_img = reshape_to_image(X_scaled[idx])
    axes[row, 0].imshow(original_img, cmap='gray')
    axes[row, 0].set_title(f'Original\n(Digit {y[idx]})')
    axes[row, 0].axis('off')
    
    # Display reconstructed images with different k values
    for col, k in enumerate(k_values, 1):
        # Reconstruct image with k components
        reconstructed = reconstruct_image(pca_full, X_scaled, k, idx)
        reconstructed_img = reshape_to_image(reconstructed)
        
        # Display reconstructed image
        axes[row, col].imshow(reconstructed_img, cmap='gray')
        axes[row, col].set_title(f'k={k}')
        axes[row, col].axis('off')

plt.tight_layout()
plt.show()

# Calculate reconstruction error (MSE) for different k values
mse_values = []
k_range = list(range(1, 51, 2))  # Test k values from 1 to 50 in steps of 2

for k in k_range:
    # Apply PCA with k components
    pca_k = PCA(n_components=k)
    X_pca_k = pca_k.fit_transform(X_scaled)
    
    # Reconstruct the entire dataset
    X_reconstructed = pca_k.inverse_transform(X_pca_k)
    
    # Calculate MSE between original and reconstructed data
    mse = mean_squared_error(X_scaled, X_reconstructed)
    mse_values.append(mse)
    print(f"k={k:2d}, MSE={mse:.6f}")

# Plot MSE vs k
plt.figure(figsize=(10, 6))
plt.plot(k_range, mse_values, 'bo-', linewidth=2, markersize=6)
plt.xlabel('Number of Principal Components (k)', fontsize=12)
plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
plt.title('Reconstruction Error vs Number of Principal Components', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1), 
         pca_full.explained_variance_ratio_, 'bo-', linewidth=2, markersize=4, label='Individual')
plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1), 
         np.cumsum(pca_full.explained_variance_ratio_), 'ro-', linewidth=2, markersize=4, label='Cumulative')
plt.xlabel('Number of Principal Components', fontsize=12)
plt.ylabel('Explained Variance Ratio', fontsize=12)
plt.title('Explained Variance Ratio vs Number of Principal Components', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print variance information
print("\nExplained variance ratio for first 10 components:")
for i in range(10):
    print(f"PC{i+1}: {pca_full.explained_variance_ratio_[i]:.4f} ({np.cumsum(pca_full.explained_variance_ratio_)[i]:.4f} cumulative)")

print(f"\nNumber of components needed for 95% variance: {np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.95) + 1}")
print(f"Number of components needed for 99% variance: {np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.99) + 1}")
