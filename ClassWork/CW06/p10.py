# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random

# Load the digits dataset
digits = load_digits()
X = digits.data  # Images flattened to 64-dimensional vectors
y = digits.target  # Corresponding labels

# Select a random image
random_idx = random.randint(0, len(X) - 1)
original_image = X[random_idx].reshape(8, 8)  # Reshape to 8x8
label = y[random_idx]

print(f"Selected image index: {random_idx}, Digit: {label}")

# Add Gaussian noise with σ²=50 (standard deviation = sqrt(50))
noise_std = np.sqrt(50)
noise = np.random.normal(0, noise_std, size=original_image.shape)
noisy_image = original_image + noise

# Clip values to be between 0 and 16 (max pixel value in digits dataset)
noisy_image = np.clip(noisy_image, 0, 16)

# Visualize original and noisy images
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Original image
axes[0].imshow(original_image, cmap='gray', vmin=0, vmax=16)
axes[0].set_title(f'Original Image (Digit: {label})')
axes[0].axis('off')

# Noisy image
axes[1].imshow(noisy_image, cmap='gray', vmin=0, vmax=16)
axes[1].set_title('Noisy Image (σ²=50)')
axes[1].axis('off')

# Prepare data for PCA
# Standardize the data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to the entire dataset
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Calculate cumulative explained variance ratio
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Find number of components that explain at least 90% of variance
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
print(f"Number of components for 90% variance: {n_components_90}")

# Apply PCA with the selected number of components
pca_90 = PCA(n_components=n_components_90)
X_pca_90 = pca_90.fit_transform(X_scaled)

# Transform the noisy image using the same PCA model
# First, we need to standardize the noisy image using the same scaler
noisy_image_flat = noisy_image.flatten().reshape(1, -1)
noisy_image_scaled = scaler.transform(noisy_image_flat)

# Transform to PCA space
noisy_pca = pca_90.transform(noisy_image_scaled)

# Reconstruct from PCA space
denoised_scaled = pca_90.inverse_transform(noisy_pca)

# Inverse the standardization
denoised = scaler.inverse_transform(denoised_scaled)

# Reshape to 8x8 and clip values
denoised_image = denoised.reshape(8, 8)
denoised_image = np.clip(denoised_image, 0, 16)

# Display denoised image
axes[2].imshow(denoised_image, cmap='gray', vmin=0, vmax=16)
axes[2].set_title(f'Denoised Image ({n_components_90} components)')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# Calculate MSE (Mean Squared Error) and PSNR (Peak Signal-to-Noise Ratio)
def calculate_mse(original, processed):
    return np.mean((original - processed) ** 2)

def calculate_psnr(original, processed, max_pixel=16):
    mse = calculate_mse(original, processed)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel / np.sqrt(mse))

mse_noisy = calculate_mse(original_image, noisy_image)
psnr_noisy = calculate_psnr(original_image, noisy_image)

mse_denoised = calculate_mse(original_image, denoised_image)
psnr_denoised = calculate_psnr(original_image, denoised_image)

print("\n" + "="*50)
print("PERFORMANCE METRICS:")
print(f"MSE (Noisy): {mse_noisy:.4f}")
print(f"PSNR (Noisy): {psnr_noisy:.2f} dB")
print(f"MSE (Denoised): {mse_denoised:.4f}")
print(f"PSNR (Denoised): {psnr_denoised:.2f} dB")
print(f"Improvement in PSNR: {psnr_denoised - psnr_noisy:.2f} dB")
print("="*50)

# Plot explained variance
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(cumulative_variance, 'b-', linewidth=2)
plt.axhline(y=0.90, color='r', linestyle='--', label='90% variance')
plt.axvline(x=n_components_90, color='g', linestyle='--', label=f'{n_components_90} components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance Ratio')
plt.legend()
plt.grid(True)

# Plot individual explained variance
plt.subplot(1, 2, 2)
plt.bar(range(1, 21), pca.explained_variance_ratio_[:20])
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Top 20 Principal Components')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
