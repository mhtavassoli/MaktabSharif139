import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read and examine the data
# Read the CSV file
df=pd.read_csv(r"E:\Maktab\Artificial Intelligence\Programming\ClassWork\CW04\p3\product_sales.csv") 

# Check data types and basic info
print("Data Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# Check if Sales column is numeric
if not np.issubdtype(df['Sales'].dtype, np.number):
    print("\nWarning: Sales column is not numeric. Converting to numeric...")
    # If 'coerce', then invalid parsing will be set as NaN.
    df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
    # Remove rows with NaN values after conversion
    df = df.dropna(subset=['Sales'])

# Step 2: Calculate mean, standard deviation and identify outliers
# Calculate mean and standard deviation
mean_sales = df['Sales'].mean()
std_sales = df['Sales'].std()

print(f"\nStatistical Summary:")
print(f"Mean Sales: {mean_sales:.2f}")
print(f"Standard Deviation: {std_sales:.2f}")

# Define outliers (more than 2 standard deviations from mean)
upper_threshold = mean_sales + 2 * std_sales
lower_threshold = mean_sales - 2 * std_sales

print(f"\nOutlier Thresholds:")
print(f"Upper Threshold (Mean + 2*STD): {upper_threshold:.2f}")
print(f"Lower Threshold (Mean - 2*STD): {lower_threshold:.2f}")

# Filter outliers
high_outliers = df[df['Sales'] > upper_threshold]
low_outliers = df[df['Sales'] < lower_threshold]
all_outliers = pd.concat([high_outliers, low_outliers])

print(f"\nOutlier Analysis:")
print(f"Number of high outliers (above {upper_threshold:.2f}): {len(high_outliers)}")
print(f"Number of low outliers (below {lower_threshold:.2f}): {len(low_outliers)}")
print(f"Total outliers: {len(all_outliers)}")

# Display outlier products
if not all_outliers.empty:
    print("\nOutlier Products:")
    print(all_outliers.sort_values('Sales', ascending=False))
else:
    print("\nNo outliers found.")

# Step 3: Visual representation with Box Plot
# plt.figure(figsize=(10, 6))

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
# Create two subplots and unpack the output array immediately

# Box plot
ax1.boxplot(df['Sales'], vert=True, patch_artist=True)
# ax1:
# It is an axes object from matplotlib
# It represents the specific subplot or plotting area where the boxplot will be drawn
# Created when using plt.subplots() or fig.add_subplot()
# vert:
# It stands for vertical Orientation
# vert=True : Boxplot is drawn vertically (default)
# vert=False: Boxplot is drawn horizontally
# patch_artist
# It refers to Box Appearance
# patch_artist=False : Boxes have only outline lines (default)
# patch_artist=True  : Boxes are filled with color

ax1.set_title('Sales Distribution - Box Plot', fontsize=14, fontweight='bold')
ax1.set_ylabel('Sales Amount', fontsize=12)
ax1.grid(True, alpha=0.3)

# Add mean line (horizontal line) and annotations 
ax1.axhline(y=mean_sales, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_sales:.2f}')
ax1.axhline(y=upper_threshold, color='orange', linestyle='--', alpha=0.7, label=f'Upper Outlier Threshold: {upper_threshold:.2f}')
ax1.axhline(y=lower_threshold, color='orange', linestyle='--', alpha=0.7, label=f'Lower Outlier Threshold: {lower_threshold:.2f}')
ax1.legend()

# Histogram with outlier regions highlighted
ax2.hist(df['Sales'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
ax2.axvline(mean_sales, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_sales:.2f}')
ax2.axvline(upper_threshold, color='orange', linestyle='--', linewidth=2, label=f'Upper Threshold: {upper_threshold:.2f}')
ax2.axvline(lower_threshold, color='orange', linestyle='--', linewidth=2, label=f'Lower Threshold: {lower_threshold:.2f}')
ax2.set_title('Sales Distribution - Histogram', fontsize=14, fontweight='bold')
ax2.set_xlabel('Sales Amount', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Highlight outlier regions
ax2.axvspan(lower_threshold, df['Sales'].min(), alpha=0.3, color='red', label='Low Outlier Region')
ax2.axvspan(upper_threshold, df['Sales'].max(), alpha=0.3, color='red', label='High Outlier Region')
# axvspan: vertical shaded region
# axhspan: horizontal shaded region

plt.tight_layout()
plt.show()

# Additional detailed box plot
plt.figure(figsize=(10, 6))
box_plot = plt.boxplot(df['Sales'], vert=True, patch_artist=True, 
                      boxprops=dict(facecolor='lightblue', color='darkblue'),
                      whiskerprops=dict(color='darkblue'),
                      capprops=dict(color='hotpink',linestyle='-.'),
                      medianprops=dict(color='red'),
                      flierprops=dict(marker='x', color='red', alpha=0.5))


# whiskerprops (Whisker Properties)
# Whiskers are the lines that extend from the box to show the range of the data
# whiskerprops = {
#     'color': 'blue',      # Line color
#     'linestyle': '--',    # Line style: '-', '--', ':', '-.'
#     'linewidth': 2,       # Line thickness
#     'alpha': 0.7,         # Transparency
# }
# capprops (Cap Properties)
# Caps are the horizontal lines at the ends of the whiskers.
# capprops = {
#     'color': 'red',       # Line color
#     'linestyle': '-',     # Line style
#     'linewidth': 3,       # Line thickness
#     'alpha': 0.8,         # Transparency
# }
#
# Visual Explanation:
# In a boxplot:
#                     Cap (capprops)
#                        ┌───┐
#                        │   │
#                Whisker │   │ (whiskerprops)
#                        │   │
#                        └───┘
#                    ┌─────────┐ ← Box
#                    │         │
#                    └─────────┘
#
# Colors:
# 🔴 Red - Bright and attention-grabbing
# 🔵 Blue - Classic and calming
# 🟢 Green - Fresh and natural
# 🟡 Yellow/Gold - Happy and energetic
# 🟠 Orange - Fun and enthusiastic
# 💖 Pink - Playful and sweet
# 🟣 Purple - Creative and magical
# 💖 hotpink 
# 💚 lime

plt.title('Product Sales - Box Plot Analysis', fontsize=14, fontweight='bold')
plt.ylabel('Sales Amount', fontsize=12)
plt.grid(True, alpha=0.3)

# Add statistical annotations
plt.text(1.1, mean_sales, f'Mean: {mean_sales:.2f}', verticalalignment='center', color='red')
plt.text(1.1, upper_threshold, f'Upper Threshold: {upper_threshold:.2f}', verticalalignment='center', color='orange')
plt.text(1.1, lower_threshold, f'Lower Threshold: {lower_threshold:.2f}', verticalalignment='center', color='orange')

plt.show()

# Step 4: Analysis and conclusions
print("\n" + "="*50)
print("ANALYSIS AND CONCLUSIONS")
print("="*50)

print(f"1. Number of outlier products: {len(all_outliers)}")
print(f"   - High outliers: {len(high_outliers)} products")
print(f"   - Low outliers: {len(low_outliers)} products")

print("\n2. Potential reasons for outliers:")
if len(high_outliers) > 0:
    print("   - High outliers might represent:")
    print("     * Best-selling products")
    print("     * Seasonal or promotional items")
    print("     * Potential data entry errors")
    
if len(low_outliers) > 0:
    print("   - Low outliers might represent:")
    print("     * Poor-performing products")
    print("     * Newly launched products")
    print("     * Products with supply issues")
    print("     * Potential data entry errors or missing data")

print("\n3. Recommended actions for further investigation:")
print("   - Verify data accuracy for outlier products")
print("   - Check for seasonal patterns or promotions")
print("   - Analyze product categories and pricing")
print("   - Investigate inventory and supply chain factors")
print("   - Consider using additional outlier detection methods (IQR, Z-score)")
print("   - Consult with sales team for business context")

# Display basic statistics
print("\n" + "="*50)
print("Sales DESCRIPTIVE STATISTICS")
print("="*50)
print(df['Sales'].describe())

# Calculate IQR for additional analysis
Q1 = df['Sales'].quantile(0.25)
Q3 = df['Sales'].quantile(0.75)
IQR = Q3 - Q1
print(f"\nIQR Analysis:")
print(f"Q1 (25th percentile): {Q1:.2f}")
print(f"Q3 (75th percentile): {Q3:.2f}")
print(f"IQR: {IQR:.2f}")
print(f"IQR Upper Bound (Q3 + 1.5*IQR): {Q3 + 1.5*IQR:.2f}")
print(f"IQR Lower Bound (Q1 - 1.5*IQR): {Q1 - 1.5*IQR:.2f}")
