import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate data for three classes
np.random.seed(42)  # For reproducible results

# Class definitions
class_params = {
    'A': {'mean': 0, 'variance': 1, 'color': 'blue', 'samples': 200},
    'B': {'mean': 3, 'variance': 0.5, 'color': 'black', 'samples': 200},
    'C': {'mean': -3, 'variance': 2, 'color': 'red', 'samples': 200}
}

# Generate data for each class
generated_data = {}
for class_name, params in class_params.items():
    generated_data[class_name] = np.random.normal(
        params['mean'], 
        np.sqrt(params['variance']), 
        params['samples']
    )

# Step 2: Combine all data into one array
all_data = np.concatenate([generated_data['A'], generated_data['B'], generated_data['C']])
true_labels = np.concatenate([
    np.full(200, 'A'),
    np.full(200, 'B'), 
    np.full(200, 'C')
])

print(f"Total data points: {len(all_data)}")
print(f"Class A: mean={np.mean(generated_data['A']):.3f}, std={np.std(generated_data['A']):.3f}")
print(f"Class B: mean={np.mean(generated_data['B']):.3f}, std={np.std(generated_data['B']):.3f}")
print(f"Class C: mean={np.mean(generated_data['C']):.3f}, std={np.std(generated_data['C']):.3f}")

# Step 3: Calculate Gaussian probability for each data point relative to each class
def gaussian_probability(x, mean, variance):
    """Calculate Gaussian probability P(x|Class)"""
    exponent = -0.5 * ((x - mean) / np.sqrt(variance))**2
    return np.exp(exponent)

# Calculate probabilities for each data point and each class
probabilities = {}
predicted_labels = []

for class_name, params in class_params.items():
    probabilities[class_name] = gaussian_probability(
        all_data, 
        params['mean'], 
        params['variance']
    )

# Step 4: Select class with maximum probability for each data point
misclassified_count = 0

for i, x in enumerate(all_data):
    class_probs = {
        'A': probabilities['A'][i],
        'B': probabilities['B'][i], 
        'C': probabilities['C'][i]
    }
    
    # Find class with maximum probability
    predicted_class = max(class_probs, key=class_probs.get)
    predicted_labels.append(predicted_class)
    
    # Check if misclassified
    if predicted_class != true_labels[i]:
        misclassified_count += 1

print(f"\nMisclassified data points: {misclassified_count}/{len(all_data)}")
print(f"Accuracy: {(1 - misclassified_count/len(all_data)) * 100:.2f}%")

# Step 5: Plot the data with colors based on predicted classes
plt.figure(figsize=(12, 6))

# Plot original data with true labels
plt.subplot(1, 2, 1)
for class_name, params in class_params.items():
    plt.scatter(
        generated_data[class_name], 
        np.zeros_like(generated_data[class_name]),
        c=params['color'], 
        label=f'Class {class_name} (True)',
        alpha=0.6
    )
plt.title('Original Data with True Labels')
plt.xlabel('x value')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot data with predicted labels
plt.subplot(1, 2, 2)
for class_name, params in class_params.items():
    # Get indices where predicted class matches current class
    indices = [i for i, label in enumerate(predicted_labels) if label == class_name]
    class_data = all_data[indices]
    
    plt.scatter(
        class_data, 
        np.zeros_like(class_data),
        c=params['color'], 
        label=f'Class {class_name} (Predicted)',
        alpha=0.6
    )
plt.title('Data with Predicted Labels (Gaussian Classification)')
plt.xlabel('x value')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional analysis: Show some misclassified examples
print("\n--- Analysis of Misclassifications ---")
misclassified_indices = [i for i in range(len(all_data)) if predicted_labels[i] != true_labels[i]]

if misclassified_indices:
    print("First 5 misclassified examples:")
    for i in misclassified_indices[:5]:
        print(f"Data point {i}: x={all_data[i]:.3f}, True={true_labels[i]}, Predicted={predicted_labels[i]}")
        print(f"  Probabilities: A={probabilities['A'][i]:.4f}, B={probabilities['B'][i]:.4f}, C={probabilities['C'][i]:.4f}")
        