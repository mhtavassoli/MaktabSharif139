import pandas as pd
import matplotlib.pyplot as plt

# 1. Define data as list of dictionaries
data = [
    {"Fruit": "Apple", "Color": "Red", "Weight": "150g", "Sweetness": 7},
    {"Fruit": "Banana", "Color": "Yellow", "Weight": "120g", "Sweetness": 9},
    {"Fruit": "Lemon", "Color": "Orange", "Weight": "100g", "Sweetness": 3}
]

# Convert to DataFrame
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
print()

# 2. Convert Weight column to numeric (remove 'g')
df['Weight'] = df['Weight'].str.replace('g', '').astype(int)
print("DataFrame after Weight conversion:")
print(df,"\n")

# 3. Visualization with matplotlib
plt.figure(figsize=(10, 6))

# Create scatter plot with actual fruit colors
colors = df['Color'].values
for _, row in df.iterrows():                    # use _ instead of i when it is not used. Also row: Series[any]
    plt.scatter(row['Sweetness'], row['Weight'], 
                color=row['Color'].lower(), 
                s=100, alpha=0.7, edgecolors='black', marker="x")
        
    # Way 1. Add fruit name labels
    plt.text(row['Sweetness']+.1,row['Weight'],
             row['Fruit'],
             fontsize=12)


# Use text() for simple text placement; It is simpler for basic labeling
# Use annotate() when you need arrows pointing to specific data points; It is more feature-rich for data explanation
# 
# plt.annotate(row['Fruit'], 
#             (row['Sweetness'], row['Weight']),
#             xytext=(10, 0), 
#             textcoords='offset points',
#             fontsize=12)
#
# xytext: Stands for X Y text.
# Purpose: It specifies the (x, y) coordinates where you want the text of the annotation (the label itself) to be placed.
# In a sentence: "This is where the text label will be located."
#
# textcoords: Stands for text coordinates.
# Purpose: It defines the coordinate system in which the xytext values are interpreted.
# 'offset points': This is a specific value for textcoords.
# It means that the values in xytext are not absolute positions on the plot,
# but rather an offset (a relative distance) measured in points
# (a typographical unit, where 1 point is 1/72 of an inch) from the point specified by the xy parameter.
#
# Combined Meaning in Simple Terms
# The code xytext=(5, 5), textcoords='offset points' means:
# "Place the annotation's text 5 points to the right and 5 points above the original data point (xy) you are annotating."
#
# Other Common Options for textcoords
# 'data': 
#         The xytext values are interpreted in the same data coordinate system as your plot 
#         (e.g., if your x-axis is from 0-10, xytext=(1, 2) would be at that specific location on the graph).
# 'figure pixels':
#                   The xytext values are pixels relative to the bottom-left of the entire figure.
# 'axes fraction': 
#                   The xytext values are fractions of the axes (from 0 to 1), 
#                   where (0, 0) is bottom-left and (1, 1) is top-right of the plot area. 
#                   This is useful for placing text in a consistent relative location.

# Way 2. Add fruit name labels
# for i in range(len(data)):
#     plt.text(data["Sweetness"].values+.2, data["Weight"].values, data["Fruit"].values)

# Chart settings
plt.xlabel('Sweetness', fontsize=12)
plt.ylabel('Weight (g)', fontsize=12)
plt.title('Feature Space Example', fontsize=14, fontweight='bold')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gcf().set_facecolor('lightblue')

# Set axis limits
plt.xlim(2.7, 9.7)
plt.ylim(97, 153)

# Add grid for better readability
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Display numerical information
print("\nNumerical fruit information:")
for i, row in df.iterrows():
    print(f"{row['Fruit']}:\t Weight = {row['Weight']}g ,\t Sweetness = {row['Sweetness']}")
    