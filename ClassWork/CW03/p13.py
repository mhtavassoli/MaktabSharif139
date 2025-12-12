import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1- Read the CSV file and explore the data
df=pd.read_csv(r"E:\Maktab\Artificial Intelligence\Programming\ClassWork\CW4\tips.csv")
print("\n",df,"\n","-"*50,"1- DataFrame")
df.info()
print(f"\nDataset shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(df.describe()) # Basic statistics

# 2- Verify that 'day' and 'tip' columns exist
requiredColumns = ['day', 'tip']
missingColumns = [col for col in requiredColumns if col not in df.columns]

if missingColumns:
    print(f"Missing columns: {missingColumns}")
    print("Available columns:", list(df.columns))
else:
    print(f"All required columns {requiredColumns} are present!")
    print(f"Unique days in dataset: {df['day'].unique()}")
    print(f"Tip range: ${df['tip'].min():.2f} to ${df['tip'].max():.2f}")

print(df[['day', 'tip']].isnull().sum()) # Check for missing values and count the number of nulls

# 3- Create Box Plot using seaborn
plt.figure(figsize=(8, 6))
boxPlot = sns.boxplot(
    x='day', 
    y='tip', 
    data=df,
    palette='Set2',  # # Set a Seaborn color palette
    showmeans=True,  # Show mean markers
    meanprops={"marker": "s", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "5"}
)
# 4- Customize the plot
plt.title('Distribution of Tips by Day of Week', fontsize=16, fontweight='bold', pad=20)
# 5- Set axis labels
plt.xlabel('Day', fontsize=12, fontweight='bold')
plt.ylabel('Tip Amount ($)', fontsize=12, fontweight='bold')

# Customize the plot
plt.grid(axis='y', alpha=0.3) 
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

# Add some statistics to the plot
dayStats = df.groupby('day')['tip'].agg(['median', 'mean']).round(2)
print(dayStats)

# Display the plot
plt.tight_layout()
plt.show()

# Calculate detailed statistics for analysis
#The aggregate() method or the agg() method, an alias of it,
# allows you to apply a function or a list of function names
# to be executed along one of the axis of the DataFrame,
# default 0, which is the index (row) axis.
# 
# ('count', 'count')   -> first is the label of column and second is the name of function
tipStats = df.groupby('day')['tip'].agg([
    ('count', 'count'),
    ('mean', 'mean'),
    ('median', 'median'),
    ('std', 'std'),
    ('min', 'min'),
    ('max', 'max')
]).round(2)
print(60*"-","\n Detailed Tip Statistics by Day:")
print(tipStats)

# Analysis Questions

# AQ1. Which day has the highest mean/median tip?
highestMeanDay = tipStats['mean'].idxmax()
highestMeanValue = tipStats['mean'].max()
highestMedianDay = tipStats['median'].idxmax()
highestMedianValue = tipStats['median'].max()
print("\n",60*"-")
print(f"   • Day with highest MEAN tip: {highestMeanDay} (${highestMeanValue})")
print(f"   • Day with highest MEDIAN tip: {highestMedianDay} (${highestMedianValue})")
print("\n",60*"-")

# AQ2. Outlier analysis
def detect_outliers(data):   # Function to detect outliers using IQR method
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers

for day in df['day'].unique():
    dayTips = df[df['day'] == day]['tip']
    outliers = detect_outliers(dayTips)
    print(f"   • {day}: {len(outliers)} outliers (Tips: {list(outliers.round(2))})")

# AQ3. Pattern analysis between weekdays and weekends

print(f"\n3. WEEKDAY VS WEEKEND PATTERNS:")
# Define weekdays and weekends
weekdays = ['Thur', 'Fri']
weekends = ['Sat', 'Sun']

weekdayTips = df[df['day'].isin(weekdays)]['tip']
weekendTips = df[df['day'].isin(weekends)]['tip']

print("\n")
print(f"   • Weekdays ({', '.join(weekdays)}):")
print(f"     - Average tip: ${weekdayTips.mean():.2f}")
print(f"     - Median tip: ${weekdayTips.median():.2f}")
print(f"     - Number of transactions: {len(weekdayTips)}")
print("\n")
print(f"   • Weekends ({', '.join(weekends)}):")
print(f"     - Average tip: ${weekendTips.mean():.2f}")
print(f"     - Median tip: ${weekendTips.median():.2f}")
print(f"     - Number of transactions: {len(weekendTips)}")

# Statistical comparison
if weekendTips.mean() > weekdayTips.mean():
    difference = weekendTips.mean() - weekdayTips.mean()
    print(f"   Average Weekend tips are ${difference:.2f} HIGHER on average than weekdays")
else:
    difference = weekdayTips.mean() - weekendTips.mean()
    print(f"   Aveeage Weekday tips are ${difference:.2f} HIGHER on average than weekends")

# Additional insights
print(f"\n4. ADDITIONAL INSIGHTS:")
print(f"   • Total days analyzed: {len(df['day'].unique())}")
print(f"   • Total transactions: {len(df)}")
print(f"   • Overall average tip: ${df['tip'].mean():.2f}")
print(f"   • Tip amount range: ${df['tip'].min():.2f} - ${df['tip'].max():.2f}")

# Day with most consistent tips (lowest standard deviation)
mostConsistentDay = tipStats['std'].idxmin()
leastConsistentDay = tipStats['std'].idxmax()
print(f"   • Most consistent tips: {mostConsistentDay} (std: ${tipStats.loc[mostConsistentDay, 'std']:.2f})")
print(f"   • Least consistent tips: {leastConsistentDay} (std: ${tipStats.loc[leastConsistentDay, 'std']:.2f})")
