import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from scipy import stats

# Section 0: Load the dataset of IRIS 

# Load the dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Display basic information about the dataset
print("Dataset Info:")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nSpecies distribution:")
print(df['species'].value_counts())

# Section 1: Calculate Statistics for Versicolor class
print("=" * 60)
print("SECTION 1: CALCULATING STATISTICS FOR VERSICOLOR")
print("=" * 60)

# Extract petal length for versicolor class
versicolor_data = df[df['species'] == 'versicolor']
petal_length_versicolor = versicolor_data['petal length (cm)']

# Calculate mean and standard deviation
mean_versicolor = np.mean(petal_length_versicolor)
std_versicolor = np.std(petal_length_versicolor)

print(f"Versicolor class statistics:")
print(f"Number of samples: {len(petal_length_versicolor)}")
print(f"Mean petal length: {mean_versicolor:.4f} cm")
print(f"Standard deviation: {std_versicolor:.4f} cm")
print(f"Minimum petal length: {np.min(petal_length_versicolor):.4f} cm")
print(f"Maximum petal length: {np.max(petal_length_versicolor):.4f} cm")
######################################################################################################## 
# Section 2: Distribution Analysis and Critical Points
print("\n" + "=" * 60)
print("SECTION 2: DISTRIBUTION ANALYSIS AND CRITICAL POINTS")
print("=" * 60)

# Calculate the 95% range using empirical rule
lower_band = mean_versicolor - 2 * std_versicolor
upper_band = mean_versicolor + 2 * std_versicolor

print(f"95% range for Versicolor petal length (using empirical rule):")
print(f"Lower band (μ - 2σ): {lower_band:.4f} cm")
print(f"Upper band (μ + 2σ): {upper_band:.4f} cm")

# Calculate actual percentage within this range
within_range = len(petal_length_versicolor[
    (petal_length_versicolor >= lower_band) & 
    (petal_length_versicolor <= upper_band)
])
actual_percentage = (within_range / len(petal_length_versicolor)) * 100

print(f"Actual percentage of Versicolor samples within this range: {actual_percentage:.2f}%")

# Create boxplot for all three classes
plt.figure(figsize=(10, 6))
sns.boxplot(x='species', y='petal length (cm)', data=df)
plt.title('Petal Length Distribution for Three Iris Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.grid(True, alpha=0.3)
plt.show()

# Section 3: Decision Boundary and Accuracy Evaluation
print("\n" + "=" * 60)
print("SECTION 3: DECISION BOUNDARY AND ACCURACY EVALUATION")
print("=" * 60)

# Define decision boundaries
decision_lower = lower_band
decision_upper = upper_band

print(f"Decision Boundary for Versicolor identification:")
print(f"Minimum petal length: {decision_lower:.4f} cm")
print(f"Maximum petal length: {decision_upper:.4f} cm")

# Evaluate accuracy
def classify_versicolor(petal_length):
    """Classify as versicolor if within decision boundaries"""
    return (petal_length >= decision_lower) & (petal_length <= decision_upper)

# Apply classification to all samples
df['predicted_versicolor'] = df['petal length (cm)'].apply(classify_versicolor)
df['actual_versicolor'] = (df['species'] == 'versicolor')

# Calculate errors
false_positives = df[
    (df['predicted_versicolor'] == True) & 
    (df['actual_versicolor'] == False)
]

false_negatives = df[
    (df['predicted_versicolor'] == False) & 
    (df['actual_versicolor'] == True)
]

print(f"\nError Analysis:")
print(f"False Positives (Type I Error): {len(false_positives)} samples")
print("These are non-Versicolor samples incorrectly classified as Versicolor:")

if len(false_positives) > 0:
    for species in false_positives['species'].unique():
        count = len(false_positives[false_positives['species'] == species])
        print(f"  - {species}: {count} samples")
        
    print("\nFalse Positive details:")
    for idx, row in false_positives.iterrows():
        print(f"  - {row['species']}: petal length = {row['petal length (cm)']:.4f} cm")
else:
    print("  - No false positives found")

print(f"\nFalse Negatives (Type II Error): {len(false_negatives)} samples")
print("These are Versicolor samples incorrectly classified as non-Versicolor:")

if len(false_negatives) > 0:
    for idx, row in false_negatives.iterrows():
        print(f"  - Sample {idx}: petal length = {row['petal length (cm)']:.4f} cm")
else:
    print("  - No false negatives found")

# Calculate accuracy metrics
total_samples = len(df)
versicolor_samples = len(versicolor_data)
correct_versicolor = len(versicolor_data) - len(false_negatives)

print(f"\nAccuracy Metrics:")
print(f"Total samples: {total_samples}")
print(f"Versicolor samples: {versicolor_samples}")
print(f"Correctly classified Versicolor: {correct_versicolor}")
print(f"Versicolor classification accuracy: {(correct_versicolor/versicolor_samples)*100:.2f}%")

# Additional visualization: Distribution with decision boundaries
plt.figure(figsize=(12, 6))

# Plot histograms for each species
species_colors = {'setosa': 'red', 'versicolor': 'blue', 'virginica': 'green'}

for species in df['species'].unique():
    species_data = df[df['species'] == species]['petal length (cm)']
    plt.hist(species_data, alpha=0.6, label=species, color=species_colors[species], bins=15)

# Add decision boundaries
plt.axvline(x=decision_lower, color='black', linestyle='--', linewidth=2, label='Decision Boundary (Lower)')
plt.axvline(x=decision_upper, color='black', linestyle='--', linewidth=2, label='Decision Boundary (Upper)')

# Add shaded area for versicolor region
plt.axvspan(decision_lower, decision_upper, alpha=0.2, color='C0', label='Versicolor Region')

plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.title('Petal Length Distribution with Versicolor Decision Boundaries')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
