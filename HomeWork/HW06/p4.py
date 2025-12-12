# ================ Part A: Load Data ================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
import ssl

# Disable SSL verification (not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context

# Load LFW dataset with minimum 60 images per person
lfw_people = fetch_lfw_people(min_faces_per_person=60, resize=0.4)

# Shape information
n_samples, h, w = lfw_people.images.shape
X = lfw_people.data  # Flattened images matrix
n_features = X.shape[1]

print(f"Number of samples (images): {n_samples}")
print(f"Number of pixels per image: {n_features}")
print(f"Original image dimensions: {h} x {w}")

# ================ Part B: PCA and Eigenfaces ================
n_components = 150
pca = PCA(n_components=n_components, whiten=True).fit(X)

# Display first 4 eigenfaces
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
for i, ax in enumerate(axes.ravel()[:4]):
    eigenface = pca.components_[i].reshape((h, w))
    ax.imshow(eigenface, cmap='gray')
    ax.set_title(f'PC{i+1}')
    ax.axis('off')
plt.suptitle('First 4 Eigenfaces')
plt.show()

'''
ENGLISH COMMENTS FOR EIGENFACES:
- PC1: Captures overall lighting variations (bright vs dark faces).
- PC2: Represents horizontal patterns (e.g., forehead vs eye regions).
- PC3 & PC4: Show vertical/diagonal structures (e.g., nose/mouth variations).
'''

# ================ Part C: Reconstruction with Different k ================
def reconstruct_image(img_vector, pca_model, k):
    """
    Reconstruct image using first k principal components
    
    Parameters:
    img_vector: Flattened image
    pca_model: Fitted PCA model
    k: Number of principal components to use
    """
    # Project to all components first
    reduced_all = pca_model.transform(img_vector.reshape(1, -1))
    
    # Keep only first k components, set others to zero
    reduced_k = reduced_all.copy()
    reduced_k[:, k:] = 0
    
    # Reconstruct from reduced components
    reconstructed = pca_model.inverse_transform(reduced_k)
    return reconstructed.reshape((h, w))

# Choose a sample image
sample_idx = 10
original_img = X[sample_idx].reshape((h, w))

# Reconstruction with different k values
k_values = [20, 80, 150]
fig, axes = plt.subplots(1, len(k_values) + 1, figsize=(12, 4))

# Original image
axes[0].imshow(original_img, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

# Reconstructed images
for i, k in enumerate(k_values):
    recon_img = reconstruct_image(X[sample_idx], pca, k)
    axes[i+1].imshow(recon_img, cmap='gray')
    axes[i+1].set_title(f'k={k}')
    axes[i+1].axis('off')
    
    # Calculate MSE
    mse = np.mean((original_img - recon_img) ** 2)
    print(f'MSE for k={k}: {mse:.4f}')

plt.suptitle('Image Reconstruction with Different k')
plt.show()

# ================ Part D: Denoising with PCA ================
# Add Gaussian noise
noise = np.random.randn(*original_img.shape) * 20  # Adjust noise level
noisy_img = original_img + noise
noisy_vector = noisy_img.flatten()

# Reconstruct using first n components
k_denoise = 100
denoised_img = reconstruct_image(noisy_vector, pca, k_denoise)

# Display
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
titles = ['Original', 'Noisy', f'Denoised (k={k_denoise})']
images = [original_img, noisy_img, denoised_img]

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
plt.show()

'''
ENGLISH COMMENTS FOR DENOISING:
- PCA reduces noise because noise is mostly in low-variance components.
- First components capture face structure; later components contain noise/unstable details.
- By truncating components, we filter out noise while preserving essential face features.
'''