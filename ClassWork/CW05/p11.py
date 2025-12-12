import pandas as pd
import numpy as np
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# 1 - Load iris dataset
print("1 - Loading iris_dataset.csv")
df=pd.read_csv(r'E:\Maktab\Artificial Intelligence\Programming\ClassWork\CW05\p11\iris_dataset.csv') 

# 2 - Separate features (X) and labels (Y)
print("2 - Separating features and labels")
X = df.iloc[:, :4]  # First 4 features
Y = df.iloc[:, 4]   # Labels

# Get unique classes
classes = Y.unique()
print(f"Classes: {classes}")

# 3 - Calculate statistics for each feature in each class
print("\n3 - Statistics for each feature in each class")
statistics = {}
for cls in classes:
    cls_data = X[Y == cls]
    stats = {
        'mean': cls_data.mean(),
        'variance': cls_data.var(),
        'range': cls_data.max() - cls_data.min(),
        'max': cls_data.max(),
        'min': cls_data.min()
    }
    statistics[cls] = stats
    print(f"\n{cls} Statistics:")
    for stat_name, values in stats.items():
        print(f"{stat_name}: {values}")

# Load test samples
print("\nLoading test samples from iris_test_samples")
# Assuming iris_test_samples is a CSV file with the same structure
test_df = pd.read_csv(r'E:\Maktab\Artificial Intelligence\Programming\ClassWork\CW05\p11\iris_test_samples.csv')
X_test = test_df.iloc[:, :4]  # First 4 features
Y_test = test_df.iloc[:, 4]   # Labels

# Use all training data for model building
X_train = X
Y_train = Y

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# 4 - Classification using only feature 3 (petal length)
print("\n4 - Classification using only petal length")
feature_idx = 2  # Feature 3 (0-indexed) - petal length
feature_name = X.columns[feature_idx]
print(f"Using feature: {feature_name}")

predictions_single = []
for i in range(len(X_test)):
    test_value = X_test.iloc[i, feature_idx]
    class_probs = {}
    
    for cls in classes:
        cls_data = X_train[Y_train == cls].iloc[:, feature_idx]
        mean = cls_data.mean()
        std = cls_data.std()
        # Avoid division by zero
        if std == 0:
            std = 1e-10
        prob = norm.pdf(test_value, mean, std)
        class_probs[cls] = prob
    
    # Predict class with highest probability
    predicted_class = max(class_probs, key=class_probs.get)
    predictions_single.append(predicted_class)

# 5 - Calculate accuracy for single feature
print("\n5 - Accuracy with single feature (petal length)")
correct_single = sum(predictions_single[i] == Y_test.iloc[i] for i in range(len(predictions_single)))
accuracy_single = 100 * correct_single / len(predictions_single)
print(f"Correct predictions: {correct_single}/{len(predictions_single)}")
print(f"Accuracy: {accuracy_single:.2f}%")

# 6 - Classification using features 3 and 4 (petal length and width)
print("\n6 - Classification using petal length and width")
feature_indices = [2, 3]  # Features 3 and 4
feature_names = [X.columns[2], X.columns[3]]
print(f"Using features: {feature_names}")

predictions_double = []
for i in range(len(X_test)):
    class_probs_avg = {}
    
    for cls in classes:
        probs = []
        for feature_idx in feature_indices:
            test_value = X_test.iloc[i, feature_idx]
            cls_data = X_train[Y_train == cls].iloc[:, feature_idx]
            mean = cls_data.mean()
            std = cls_data.std()
            if std == 0:
                std = 1e-10
            prob = norm.pdf(test_value, mean, std)
            probs.append(prob)
        
        # Average probability of both features
        class_probs_avg[cls] = np.mean(probs)
    
    # Predict class with highest average probability
    predicted_class = max(class_probs_avg, key=class_probs_avg.get)
    predictions_double.append(predicted_class)

# 7 - Calculate accuracy for two features
print("\n7 - Accuracy with two features (petal length and width)")
correct_double = sum(predictions_double[i] == Y_test.iloc[i] for i in range(len(predictions_double)))
accuracy_double = 100 * correct_double / len(predictions_double)
print(f"Correct predictions: {correct_double}/{len(predictions_double)}")
print(f"Accuracy: {accuracy_double:.2f}%")

# 8 - Classification using all 4 features with top 2 probabilities
print("\n8 - Classification using all 4 features (top 2 probabilities)")
predictions_quad = []
for i in range(len(X_test)):
    class_probs_top2 = {}
    
    for cls in classes:
        probs = []
        for feature_idx in range(4):  # All 4 features
            test_value = X_test.iloc[i, feature_idx]
            cls_data = X_train[Y_train == cls].iloc[:, feature_idx]
            mean = cls_data.mean()
            std = cls_data.std()
            if std == 0:
                std = 1e-10
            prob = norm.pdf(test_value, mean, std)
            probs.append(prob)
        
        # Take top 2 probabilities and average them
        top2_probs = sorted(probs, reverse=True)[:2]
        class_probs_top2[cls] = np.mean(top2_probs)
    
    # Predict class with highest average of top 2 probabilities
    predicted_class = max(class_probs_top2, key=class_probs_top2.get)
    predictions_quad.append(predicted_class)

# Calculate accuracy for 4 features with top 2
correct_quad = sum(predictions_quad[i] == Y_test.iloc[i] for i in range(len(predictions_quad)))
accuracy_quad = 100 * correct_quad / len(predictions_quad)
print(f"Correct predictions: {correct_quad}/{len(predictions_quad)}")
print(f"Accuracy: {accuracy_quad:.2f}%")

# Detailed results comparison
print("\n" + "="*60)
print("DETAILED RESULTS COMPARISON")
print("="*60)
print(f"{'Method':<40} {'Accuracy':<10} {'Correct/Total'}")
print("-" * 60)
print(f"{'Single feature (petal length)':<40} {accuracy_single:<10.2f}% {correct_single}/{len(predictions_single)}")
print(f"{'Two features (petal length & width)':<40} {accuracy_double:<10.2f}% {correct_double}/{len(predictions_double)}")
print(f"{'Four features (top 2 probabilities)':<40} {accuracy_quad:<10.2f}% {correct_quad}/{len(predictions_quad)}")

# Show some test samples with predictions
print("\nSAMPLE TEST PREDICTIONS (first 10 samples):")
print("Index | Actual Class | Single Feature | Two Features | Four Features")
print("-" * 65)
for i in range(min(10, len(X_test))):
    print(f"{i:5} | {Y_test.iloc[i]:<12} | {predictions_single[i]:<14} | {predictions_double[i]:<12} | {predictions_quad[i]}")