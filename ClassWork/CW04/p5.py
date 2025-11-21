import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Create new data for each fruit
# Original data from previous question
original_data = [
    {'Fruit': 'Apple', 'Color': 'red', 'Weight': 150, 'Sweetness': 7},
    {'Fruit': 'Banana', 'Color': 'yellow', 'Weight': 120, 'Sweetness': 9},
    {'Fruit': 'Lemon', 'Color': 'orange', 'Weight': 100, 'Sweetness': 3}
]

# Create new samples for each fruit with realistic variations
new_samples = []
for fruit in ['Apple', 'Banana', 'Lemon']:
    if fruit == 'Apple':
        # Create 3 new apple samples with slight variations
        new_samples.extend([
            {'Fruit': 'Apple', 'Color': 'red', 'Weight': 145, 'Sweetness': 7.5},
            {'Fruit': 'Apple', 'Color': 'red', 'Weight': 155, 'Sweetness': 6.8},
            {'Fruit': 'Apple', 'Color': 'red', 'Weight': 148, 'Sweetness': 7.2}
        ])
    elif fruit == 'Banana':
        # Create 3 new banana samples with slight variations
        new_samples.extend([
            {'Fruit': 'Banana', 'Color': 'yellow', 'Weight': 118, 'Sweetness': 8.9},
            {'Fruit': 'Banana', 'Color': 'yellow', 'Weight': 125, 'Sweetness': 9.1},
            {'Fruit': 'Banana', 'Color': 'yellow', 'Weight': 122, 'Sweetness': 8.7}
        ])
    elif fruit == 'Lemon':
        # Create 3 new lemon samples with slight variations
        new_samples.extend([
            {'Fruit': 'Lemon', 'Color': 'orange', 'Weight': 95, 'Sweetness': 3.2},
            {'Fruit': 'Lemon', 'Color': 'orange', 'Weight': 105, 'Sweetness': 2.9},
            {'Fruit': 'Lemon', 'Color': 'orange', 'Weight': 98, 'Sweetness': 3.4}
        ])

# Step 2: Add new data to the original dataframe
# Create DataFrames
df_original = pd.DataFrame(original_data)
df_new = pd.DataFrame(new_samples)

# Add a column to identify original vs new data
df_original['Data_Type'] = 'Original'
df_new['Data_Type'] = 'New'

# Combine both DataFrames
df_combined = pd.concat([df_original, df_new], ignore_index=True)

print("Combined DataFrame:")
print(df_combined)

# Step 3: Create the final scatter plot
plt.figure(figsize=(10, 6))

# Define color mapping for fruits
color_map = {
    'red': '#FF6B6B',      # Light red for apples
    'yellow': "#F2FF00",    # Light yellow for bananas
    'orange': "#FFB83DDC",    # Light yellow for bananas
}

# Plot original data with 'x' marker
original_data = df_combined[df_combined['Data_Type'] == 'Original']
for fruit in original_data['Fruit'].unique():
    fruit_data = original_data[original_data['Fruit'] == fruit]
    plt.scatter(fruit_data['Sweetness'], 
                fruit_data['Weight'], 
                marker='x', 
                s=100,  # Size of markers
                c=color_map[fruit_data['Color'].iloc[0]],
                label=f'{fruit} (Original)',
                edgecolors='black',
                linewidth=2)

# Plot new data with triangle marker '^'
new_data = df_combined[df_combined['Data_Type'] == 'New']
for fruit in new_data['Fruit'].unique():
    fruit_data = new_data[new_data['Fruit'] == fruit]
    plt.scatter(fruit_data['Sweetness'], 
                fruit_data['Weight'], 
                marker='^',  # Triangle marker
                s=100,  # Size of markers
                c=color_map[fruit_data['Color'].iloc[0]],
                label=f'{fruit} (New)',
                edgecolors='black',
                alpha=0.8)

# Add fruit names as labels for all data points
for i, row in df_combined.iterrows():
    plt.annotate(row['Fruit'], 
                xy=(row['Sweetness'], row['Weight']),
                xytext=(8, 0),  # Offset for text position
                textcoords='offset points',
                fontsize=9,
                alpha=0.8)

# Customize the plot
plt.xlabel('Sweetness Level', fontsize=12, fontweight='bold')
plt.ylabel('Weight (grams)', fontsize=12, fontweight='bold')
plt.title('Fruit Analysis: Weight vs Sweetness (Original vs New Samples)', 
          fontsize=14, fontweight='bold', pad=20)

# Remove top and right spines for cleaner look
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Add grid for better readability
plt.grid(True, alpha=0.3, linestyle='--')

# Create custom legend for data types
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='x', color='black', label='Original Data',
           markerfacecolor='gray', markersize=10, linestyle='None'),
    Line2D([0], [0], marker='^', color='black', label='New Data',
           markerfacecolor='gray', markersize=10, linestyle='None')
]

# Add both legends (fruit types and data types)
plt.legend(loc='upper left', frameon=True, fancybox=True, 
           shadow=True, framealpha=0.9)

# Adjust layout and display
plt.tight_layout()
plt.show()

# Display some statistics
print(f"\nDataset Statistics:")
print(f"Total samples: {len(df_combined)}")
print(f"Original samples: {len(df_original)}")
print(f"New samples: {len(df_new)}")
print(f"\nFruit distribution:")
print(df_combined['Fruit'].value_counts())