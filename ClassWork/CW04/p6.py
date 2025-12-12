import pandas as pd
import matplotlib.pyplot as plt

# Create the fruits dataset
fruits_data = {
    'Fruit': ['Apple', 'Banana', 'Lemon', 'Orange', 'Strawberry', 'Grape', 'Watermelon', 'Pineapple', 'Kiwi'],
    'Weight': [150, 120, 100, 200, 25, 5, 5000, 1000, 75],
    'Sweetness': [7, 9, 3, 6, 8, 9, 7, 8, 6],
    'Color': ['red', 'yellow', 'orange', 'orange', 'red', 'purple', 'green', 'yellow', 'brown']
}

df = pd.DataFrame(fruits_data)

# --- Step 1: Feature Engineering ---
# Add a new feature: Juiciness (scale 1-10)
juiciness_values = {
    'Apple': 6,
    'Banana': 5,
    'Orange': 9,
    'Strawberry': 8,
    'Grape': 7,
    'Watermelon': 9,
    'Pineapple': 8,
    'Kiwi': 8
}

# Add the new feature to the dataframe
df['Juiciness'] = df['Fruit'].map(juiciness_values)

print("Dataset with new feature:")
print(df)

# --- Step 2: 3D Visualization ---

# Create color mapping for the fruits
color_mapping = {
    'red': 'red',
    'yellow': 'yellow', 
    'orange': 'orange',
    'purple': 'purple',
    'green': 'green',
    'brown': 'brown'
}

# Map color names to actual colors for plotting
df['Color_Code'] = df['Color'].map(color_mapping)

# Create 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot each fruit with its actual color
for color_name in df['Color'].unique():
    # Filter data for current color
    color_data = df[df['Color'] == color_name]
    
    # Plot points for this color
    ax.scatter(color_data['Sweetness'], 
               color_data['Weight'], 
               color_data['Juiciness'],
               c=color_data['Color_Code'],
               label=color_name,
               s=100,  # Size of points
               alpha=0.7)
    
# Add fruit names as text labels next to each point
for i, row in df.iterrows():
    ax.text(row['Sweetness'] - 0.1,  # X position with small offset
            row['Weight'] + 50,      # Y position with small offset  
            row['Juiciness'] + 0.1,  # Z position with small offset
            row['Fruit'],            # Fruit name
            fontsize=10,
            fontweight='bold',
            ha='left',               # Horizontal alignment
            va='bottom',             # Vertical alignment
            bbox=dict(boxstyle='round,pad=0.3', 
                     facecolor='white', 
                     alpha=0.8,
                     edgecolor='gray'))


# Add labels and title
ax.set_xlabel('Sweetness', fontsize=12, labelpad=10, fontweight= 'bold')
ax.set_ylabel('Weight (grams)', fontsize=12, labelpad=10, fontweight= 'bold')
ax.set_zlabel('Juiciness (1-10)', fontsize=12, labelpad=10, fontweight= 'bold')
ax.set_title('3D Visualization of Fruits: Sweetness vs Weight vs Juiciness', fontsize=14, pad=20)

# Add legend
ax.legend(title='Fruit Colors', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust viewing angle for better perspective
ax.view_init(elev=20, azim=45)

# Add grid for better readability
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Optional: Display the final dataset
print("\nFinal dataset with all features:")
print(df[['Fruit', 'Weight', 'Sweetness', 'Juiciness', 'Color']])


""" way 2
import pandas as pd
import matplotlib.pyplot as plt

data = [
{"Fruit": "Apple", "Color": "red", "Weight": "150g", "Sweetness": 7},
{"Fruit": "Banana", "Color": "yellow", "Weight": "120g", "Sweetness": 8},
{"Fruit": "Lemon", "Color": "orange", "Weight": "80g", "Sweetness": 3}
]
new_data = [{"Fruit": "Apple", "Color": "red", "Weight": "140g", "Sweetness": 6},
{"Fruit": "Banana", "Color": "yellow", "Weight": "130g", "Sweetness": 9},
{"Fruit": "Lemon", "Color": "orange", "Weight": "75g", "Sweetness": 4}
]
data = pd.DataFrame(data)
new_data = pd.DataFrame(new_data)
finall_data = pd.concat([data,new_data],ignore_index=True)
print(finall_data)

plt.scatter(data["Sweetness"],data["Weight"],c=data["Color"],marker= 'x')
plt.xlabel("Sweetness")
plt.ylabel("Weight")
for i in range(len(data)):
    plt.text(data["Sweetness"][i],data["Weight"][i],data["Fruit"][i])

plt.scatter(new_data["Sweetness"],new_data["Weight"],c=new_data["Color"],marker= '*')
plt.xlabel("Sweetness")
plt.ylabel("Weight")

for i in range(len(new_data)):
    plt.text(new_data["Sweetness"][i],new_data["Weight"][i],new_data["Fruit"][i])

plt.legend(["old","new"])
plt.show()
"""
