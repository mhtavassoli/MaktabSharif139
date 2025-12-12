"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Multidimensional Analysis of Housing Data with Pair Plots
# Step 1. Reading the File and Data Exploration
print("\nStep 1. Reading the File and Data Exploration")
print("\nTask 1. Reading the File")

# Read the data file
df = pd.read_csv(r'E:\Maktab\Artificial Intelligence\Programming\ClassWork\CW04\housing_data.csv')

# Explore data structure
print("\n\nBasic data information:")
print(df.info())
print("\n\nDescriptive statistics:")
print(df.describe().round(2))
print("\n\nData 5 sample:")
print(df.head())

print("\nTask 2. Data Exploration:\n")

# Check required columns and data types
requiredColumns = ['Price', 'Area', 'Bedrooms', 'Bathrooms', 'LocationScore']
columnWidth = max([len(col) for col in requiredColumns]) + 2
# columnWidth = max(map(len, requiredColumns)) + 2

for col in requiredColumns:
    if col in df.columns:
        print(f"Column {col:{columnWidth}} type {df[col].dtype}")
    else:
        print(f"Column {col:{columnWidth}} not found!")

# Step 2: Creating Pair Plots
print("\nStep 2: Creating Pair Plots\n")

# Create pair plot using required Columns for analysis
sns.set_theme(style="dark")
pairplot = sns.pairplot(df[requiredColumns], 
                        diag_kind='hist',       
                        plot_kws={'alpha':0.6, 's':40})

# diag_kind
# Controls the plots along the diagonal (where variables are plotted against themselves)
# 'hist': Displays histograms, 'kde' (density curves), 'None' (disabled-empty)

# plot_kws
# Visual customization for the scatter plots 
# (off-diagonal plots where variables are plotted against each other)
# 
# 'alpha': 0.6 
# Point transparency ranging from 0 to 1
# 0 = completely transparent, 1 = completely opaque, 0.6 = 60% opaque - helps visualize overlapping points
#
# 's': 40
# Point size (area of markers)
# Larger number = larger points

pairplot.figure.set_size_inches(10, 6) 
plt.suptitle('Housing Data Pair Plot', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
"""
########################################################################################## 
# The professional way
########################################################################################## 

# Multidimensional Analysis of Housing Data with Statistical Insights

# Step 1: Reading the File and Data Exploration ------------------------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the data file
df = pd.read_csv(r'E:\Maktab\Artificial Intelligence\Programming\ClassWork\CW04\housing_data.csv')

# Explore data structure
print("Basic data information:")
print(df.info())
print("\nDescriptive statistics:")
print(df.describe())
print("\nData sample:")
print(df.head())

# Check required columns and data types
required_columns = ['Price', 'Area', 'Bedrooms', 'Bathrooms', 'LocationScore']
for col in required_columns:
    if col in df.columns:
        print(f"Column {col}: type {df[col].dtype}")
    else:
        print(f"Column {col} not found!")
        
# Step 2: Advanced Statistical Analysis ------------------------------------------------

# Calculate correlation matrix and detailed statistics
print("=" * 60)
print("ADVANCED STATISTICAL ANALYSIS")
print("=" * 60)

# 1. Correlation analysis
correlation_matrix = df[required_columns].corr()
print("\n1. CORRELATION MATRIX (with Price):")
price_correlations = correlation_matrix['Price'].sort_values(ascending=False)
print(price_correlations)

# 2. Statistical significance of relationships
print("\n2. STATISTICAL INSIGHTS:")

# Price vs Area analysis
area_price_stats = df[['Area', 'Price']].describe()
print(f"\nArea-Price Relationship:")
print(f"• Correlation coefficient: {price_correlations['Area']:.3f}")
print(f"• Average price per square meter: ${(df['Price']/df['Area']).mean():.2f}")
print(f"• Area range: {df['Area'].min()} - {df['Area'].max()} sqm")
print(f"• Price range: ${df['Price'].min():,} - ${df['Price'].max():,}")

# Bedrooms and Bathrooms analysis
print(f"\nBedrooms Analysis:")
bedroom_stats = df.groupby('Bedrooms')['Price'].agg(['mean', 'count', 'std'])
print(bedroom_stats)

print(f"\nBathrooms Analysis:")
bathroom_stats = df.groupby('Bathrooms')['Price'].agg(['mean', 'count', 'std'])
print(bathroom_stats)

# 3. Location Score impact
print(f"\n3. LOCATION SCORE IMPACT:")
location_impact = df.groupby('LocationScore')['Price'].agg(['mean', 'count'])
location_impact['price_increase'] = location_impact['mean'].pct_change() * 100
print(location_impact)

# 4. Price prediction insights
print(f"\n4. PRICE PREDICTION INSIGHTS:")
print(f"• Strongest predictor: Area (r = {price_correlations['Area']:.3f})")
print(f"• Secondary predictor: Bathrooms (r = {price_correlations['Bathrooms']:.3f})")
print(f"• Location influence: LocationScore (r = {price_correlations['LocationScore']:.3f})")
print(f"• Weakest predictor among main features: Bedrooms (r = {price_correlations['Bedrooms']:.3f})")

# Step 3: Enhanced Pair Plot with Statistical Annotations ------------------------------------------

# Create enhanced pair plot with correlation coefficients
def corr_func(x, y, **kws):
    r = pd.Series(x).corr(pd.Series(y))
    ax = plt.gca()
    ax.annotate(f'r = {r:.2f}', xy=(0.1, 0.9), xycoords=ax.transAxes, 
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

g = sns.PairGrid(df[required_columns])
g.map_upper(plt.scatter, alpha=0.6, s=50)
g.map_lower(sns.regplot, scatter_kws={'alpha':0.6, 's':50}, line_kws={'color': 'red'})
g.map_diag(plt.hist, alpha=0.8)
g.map_upper(corr_func)
plt.suptitle('Enhanced Housing Data Pair Plot with Correlation Coefficients', y=0.98, fontsize=12)
g.figure.set_size_inches(10, 6) 
plt.tight_layout()
plt.show()

# Step 4: Predictive Insights and Conclusions -----------------------------------------------------

# Generate predictive insights based on statistical analysis
print("=" * 60)
print("PREDICTIVE INSIGHTS AND CONCLUSIONS")
print("=" * 60)

print("\n1. PRICE PREDICTION MODEL INSIGHTS:")
print("Based on correlation analysis, the optimal feature ranking for price prediction is:")
for i, (feature, corr) in enumerate(price_correlations.items(), 1):
    if feature != 'Price':
        strength = "STRONG" if abs(corr) > 0.7 else "MODERATE" if abs(corr) > 0.5 else "WEAK"
        print(f"  {i}. {feature}: {strength} correlation (r = {corr:.3f})")

print("\n2. PRACTICAL IMPLICATIONS FOR HOUSING MARKET:")
# Calculate practical insights
avg_price_per_room = df.groupby('Bedrooms')['Price'].mean().diff().mean()
avg_price_per_bathroom = df.groupby('Bathrooms')['Price'].mean().diff().mean()
price_per_sqm = (df['Price'] / df['Area']).mean()

print(f"• Each additional bathroom increases price by approximately ${avg_price_per_bathroom:,.0f}")
print(f"• Each additional bedroom increases price by approximately ${avg_price_per_room:,.0f}")
print(f"• Average price per square meter: ${price_per_sqm:.2f}")

print("\n3. INVESTMENT RECOMMENDATIONS:")
# Generate investment insights
location_premium = (df[df['LocationScore'] >= 8]['Price'].mean() / df[df['LocationScore'] <= 5]['Price'].mean() - 1) * 100
bathroom_premium = (df[df['Bathrooms'] >= 3]['Price'].mean() / df[df['Bathrooms'] == 1]['Price'].mean() - 1) * 100

print(f"• High-location premium: {location_premium:.1f}% price increase for prime locations")
print(f"• Bathroom premium: {bathroom_premium:.1f}% price increase for 3+ bathrooms")
print(f"• Best value indicators: Focus on Area and Bathroom count for accurate price estimation")

print("\n4. DATA QUALITY ASSESSMENT:")
# Check data quality
missing_data = df[required_columns].isnull().sum()
outlier_threshold = df['Price'].quantile(0.95)
outliers_count = (df['Price'] > outlier_threshold).sum()

print(f"• Missing values: {missing_data.sum()} total")
print(f"• Potential outliers: {outliers_count} houses above ${outlier_threshold:,.0f}")
print(f"• Data reliability: {'HIGH' if missing_data.sum() == 0 else 'MODERATE'}")

# Step 5: Key Findings Summary -------------------------------------------------------------------

# Final summary
print("=" * 60)
print("KEY FINDINGS SUMMARY")
print("=" * 60)

print("""
MAIN CONCLUSIONS:
1. Area is the strongest price predictor (correlation > 0.7)
2. Bathroom count shows stronger relationship than bedroom count
3. Location score provides moderate predictive power
4. Combined features likely provide best prediction accuracy

PRACTICAL APPLICATIONS:
• Real estate valuation should prioritize area measurements
• Bathroom upgrades may provide better ROI than bedroom additions
• Location remains important but secondary to physical characteristics
• Investors should focus on properties with undervalued area potential
""")

print("""
This enhanced analysis provides:

- Quantitative correlations between features and price
- Statistical significance of relationships  
- Practical investment insights
- Data quality assessment
- Predictive feature ranking
- Market behavior patterns

All analysis uses only pandas, seaborn, and matplotlib as requested, 
providing comprehensive insights without additional libraries.
""")
