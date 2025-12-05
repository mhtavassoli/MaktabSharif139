import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

# 1. بارگذاری دیتاست
diabetes = load_diabetes()
X = diabetes.data
feature_names = diabetes.feature_names
df = pd.DataFrame(X, columns=feature_names)

print("Dataset shape:", X.shape)
print("\nFeature names:", feature_names)

# 2. محاسبه ماتریس همبستگی
correlation_matrix = df.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# 3. شناسایی ویژگی‌های با همبستگی بالا (|corr| > 0.5)
high_corr_features = []
for i in range(len(feature_names)):
    for j in range(i+1, len(feature_names)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > 0.5:
            high_corr_features.append((feature_names[i], feature_names[j], corr_val))
            print(f"High correlation: {feature_names[i]} - {feature_names[j]}: {corr_val:.3f}")

# استخراج ویژگی‌های منحصر به فرد با همبستگی بالا
unique_high_corr_features = set()
for feat1, feat2, _ in high_corr_features:
    unique_high_corr_features.add(feat1)
    unique_high_corr_features.add(feat2)

unique_high_corr_features = list(unique_high_corr_features)
print(f"\nFeatures with |correlation| > 0.5: {unique_high_corr_features}")

# 4. اجرای PCA روی زیرمجموعه ویژگی‌های با همبستگی بالا
subset_df = df[unique_high_corr_features]

# استانداردسازی داده‌ها
scaler = StandardScaler()
X_scaled = scaler.fit_transform(subset_df)

# اجرای PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# بارهای مؤلفه‌های اصلی
loadings = pca.components_
print("\nPCA Loadings for PC1 and PC2:")
for i, feature in enumerate(unique_high_corr_features):
    print(f"{feature}: PC1={loadings[0, i]:.3f}, PC2={loadings[1, i]:.3f}")

# 5. تجزیه و تحلیل خوشه‌بندی ویژگی‌های همبسته
print("\nAnalysis of feature clustering in principal components:")
for i, pc in enumerate(['PC1', 'PC2']):
    print(f"\n{pc} dominant features (|loading| > 0.5):")
    for j, feature in enumerate(unique_high_corr_features):
        if abs(loadings[i, j]) > 0.5:
            print(f"  {feature}: {loadings[i, j]:.3f}")

# 6. انتخاب دو نمونه تصادفی و محاسبه فاصله ماهالانوبیس
np.random.seed(42)
sample_indices = np.random.choice(len(X), 2, replace=False)
sample1 = X_scaled[sample_indices[0]]
sample2 = X_scaled[sample_indices[1]]

print(f"\nSelected samples indices: {sample_indices}")

# محاسبه فاصله ماهالانوبیس در فضای اصلی
cov_matrix = np.cov(X_scaled.T)
inv_cov_matrix = np.linalg.pinv(cov_matrix)  # استفاده از pseudoinverse برای پایداری عددی
diff = sample1 - sample2
mahalanobis_dist_original = np.sqrt(diff.T @ inv_cov_matrix @ diff)
print(f"Mahalanobis distance in original space: {mahalanobis_dist_original:.4f}")

# محاسبه فاصله ماهالانوبیس در فضای PCA
# در فضای PCA، کوواریانس ماتریس قطری است (مؤلفه‌ها ناهمبسته هستند)
pca_diff = X_pca[sample_indices[0]] - X_pca[sample_indices[1]]
# واریانس هر مؤلفه اصلی
pca_variances = pca.explained_variance_
# فاصله ماهالانوبیس در فضای PCA (با فرض نرمال بودن)
mahalanobis_dist_pca = np.sqrt(np.sum((pca_diff**2) / pca_variances))
print(f"Mahalanobis distance in PCA space: {mahalanobis_dist_pca:.4f}")

# 7. فاصله اقلیدسی برای مقایسه
euclidean_dist_original = np.linalg.norm(sample1 - sample2)
euclidean_dist_pca = np.linalg.norm(pca_diff)
print(f"\nEuclidean distance in original space: {euclidean_dist_original:.4f}")
print(f"Euclidean distance in PCA space: {euclidean_dist_pca:.4f}")

# 8. مصورسازی
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Heatmap همبستگی
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
            center=0, ax=axes[0, 0])
axes[0, 0].set_title('Correlation Matrix of All Features')

# Heatmap همبستگی زیرمجموعه
subset_corr = subset_df.corr()
sns.heatmap(subset_corr, annot=True, fmt=".2f", cmap='coolwarm', 
            center=0, ax=axes[0, 1])
axes[0, 1].set_title('Correlation Matrix of High-Correlation Features')

# بارهای مؤلفه‌های اصلی
x = np.arange(len(unique_high_corr_features))
width = 0.35
axes[1, 0].bar(x - width/2, loadings[0], width, label='PC1')
axes[1, 0].bar(x + width/2, loadings[1], width, label='PC2')
axes[1, 0].set_xlabel('Features')
axes[1, 0].set_ylabel('Loadings')
axes[1, 0].set_title('PCA Loadings for PC1 and PC2')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(unique_high_corr_features, rotation=45)
axes[1, 0].legend()
axes[1, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
axes[1, 0].axhline(y=-0.5, color='r', linestyle='--', alpha=0.5)

# نمودار پراکنش در فضای PCA
axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
axes[1, 1].scatter(X_pca[sample_indices[0], 0], X_pca[sample_indices[0], 1], 
                   color='red', s=100, label=f'Sample {sample_indices[0]}')
axes[1, 1].scatter(X_pca[sample_indices[1], 0], X_pca[sample_indices[1], 1], 
                   color='green', s=100, label=f'Sample {sample_indices[1]}')
axes[1, 1].set_xlabel('PC1 (Variance: {:.1f}%)'.format(pca.explained_variance_ratio_[0]*100))
axes[1, 1].set_ylabel('PC2 (Variance: {:.1f}%)'.format(pca.explained_variance_ratio_[1]*100))
axes[1, 1].set_title('Samples in PCA Space')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# 9. اطلاعات PCA
print("\nPCA Explained Variance Ratios:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio:.3f} ({ratio*100:.1f}%)")

print(f"\nCumulative variance explained by first 2 PCs: {np.sum(pca.explained_variance_ratio_[:2]):.3f}")
