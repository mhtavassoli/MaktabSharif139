import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1. Reading the file and preparing the data

# Task 1. Read the CSV file and Store it in a Dataframe
df=pd.read_csv(r"E:\Maktab\Artificial Intelligence\Programming\ClassWork\CW04\p1\daily_sales.csv") 
print("\n",df,"\n","-"*50,"1- DataFrame")

# Task 2. Check that the file contains the following two columns:
    # Date: Sales date (in DD-MM-YYYY format)
    # Sales: Daily sales amount (integer)
print("Columns in the DataFrame:")
print(df.columns.tolist())
print("\nDataFrame info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())

"""
if "df" not in df.columns:
    print("Date Column Does Not Exist")
if "Sales" not in df.columns:
    print("Sales Column Does Not Exist")
"""

# Task 3. Convert the Date column to datetime type
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# Task 4. Set the Date column as the DataFrame index
df.set_index('Date', inplace=True)

print(df.head())

"""
df["Date"]=pd.to_datetime(df["Date"])
df.index=df["Date"]
data=df.drop("Date",axis=1)
print(data)
"""
# Step 2. Calculate the 7-day moving average

# Task 1&2. The result is already added as a new column named 'RollingMean'
# df['RollingMean'] = df['Sales'].rolling(window=7,min_periods=3).mean()

# Task 1. Calculate the 7-day moving average for Sales column using .rolling()
rolling_mean = df['Sales'].rolling(window=7, min_periods=3).mean()
type(rolling_mean)

# Task 2. Add the result as a new column named 'RollingMean' to the DataFrame
df['RollingMean'] = rolling_mean

print(df.head(10))

"""
em7=data.rolling(7,min_periods=3).mean()
em7=em7.rename(columns={"Sales": "Roll"})
# print(em7)
data=pd.concat([data,em7],axis=1)
"""
# Step 3. Draw the chart

# Task 1. Create a line plot using matplotlib
plt.figure(figsize=(11, 6))
# plt.plot(df.index, df['Sales'], color='blue', label='Daily Sales', linewidth=1)
plt.plot(df.index, df['Sales'], label='Daily Sales', linewidth=1)

# Task 2.1. Plot original sales with blue line
plt.gca().get_lines()[-1].set_color('blue') 
# [-1] refers to the last line added to the chart
# gca refer to abbreviated form of Get Current Axes

# Task 2.2. Plot moving average with red line
plt.plot(df.index, df['RollingMean'], color='red', label='7-Day Moving Average', linewidth=2)

# 3. Set chart title
plt.title('Daily Sales vs 7-Day Moving Average', fontsize=14, fontweight='bold')

# 4. Label X and Y axes and add legend
plt.xlabel('Date', fontsize=12, fontweight='bold')
plt.ylabel('Sales Amount', fontsize=12, fontweight='bold')
plt.legend(bbox_to_anchor=(1.0, -0.05), loc='upper right', fontsize=8, framealpha=0.9)
# Add legend outside the plot at the bottom right
# 1.0: Positions the legend at the right edge of the plot
# -0.05: Positions the legend 15% below the bottom of the plot
# loc='upper right': Anchors the legend's upper-right corner to the bbox_to_anchor position
# framealpha=0.9: Sets transparency of legend background
# This places the legend outside and below the plot area at the bottom-right corner.

# Display the plot
plt.grid(True, alpha=0.3)
# Function: Adds grid lines to the plot
    # True: Enables grid lines
    # alpha=0.3: Sets transparency level (0.0 = completely transparent, 1.0 = completely opaque)
        # Makes grid lines faint so they don't overpower the data
# Result: Light gray grid lines in the background for easier reading of values

plt.show()

# Step 4. Data analysis

# Task 1. - In which time periods did sales increase or decrease suddenly?
print("\nTask 1. Identifying periods with sudden increases or decreases:")
df['Deviation'] = abs(df['Sales'] - df['RollingMean']) # Calculate deviation from moving average
df['PctDeviation'] = (df['Deviation'] / df['RollingMean']) * 100 # Percent Deviation

threshold = 50 # Setting threshold for significant fluctuations (e.g., deviation more than 50%)

significantDeviations = df[df['PctDeviation'] > threshold]
if not significantDeviations.empty:
    print(f"Number of days with significant fluctuations (deviation > {threshold}%): {len(significantDeviations)}")
    
    print("\nLargest negative fluctuations:")
    negativeSpikes = df[df['Sales'] < df['RollingMean']].nlargest(5, 'PctDeviation')
    for date, row in negativeSpikes.iterrows():
        print(f"  {date.strftime('%Y-%m-%d')}: Sales {row['Sales']} (deviation {row['PctDeviation']:.1f}%)")
    
    print("\nLargest positive fluctuations:")
    positiveSpikes = df[df['Sales'] > df['RollingMean']].nlargest(5, 'PctDeviation')
    for date, row in positiveSpikes.iterrows():
        print(f"  {date.strftime('%Y-%m-%d')}: Sales {row['Sales']} (deviation {row['PctDeviation']:.1f}%)")
else:
    print("No significant fluctuations identified.")

# Task 2. Is the overall sales trend up or down in the given time period?
print("\nTask 2. Overall sales trend in the given time period analysis:")

# Analyzing overall sales trend using linear regression consept (Manual calculation) on moving average

# Remove NaN values from moving average
rollingData = df[['RollingMean']].dropna()
if len(rollingData) > 1:
    # Manual linear regression calculation
    x = np.arange(len(rollingData)) #     x = np.array(range(len(rolling_data))).reshape(-1, 1)
    y = rollingData['RollingMean'].values
    
    # Calculate slope manually using least squares method
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    if denominator != 0:
        slope = numerator / denominator
    else:
        slope = 0

    """""
    # Calculate trend using linear regression on moving average
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    """""    
    
    # Way 1- Analyze the trend based on slope
    if slope > 0:
        trend = "Upward"
    elif slope < 0:
        trend = "Downward"
    else:
        trend = "Stable"
    print("\nWay 1- Analyze the trend based on slope")
    print(f"\n   - Overall trend: {trend} (slope: {slope:.6f})")
    
    # Way 2- Analyze the trend by Comparing first 30% vs last 30% of the period   
    splitPoint = int(len(rollingData) * 0.3)
    
    if splitPoint > 0:
        firstPeriodMean = rollingData['RollingMean'].iloc[:splitPoint].mean()
        lastPeriodMean = rollingData['RollingMean'].iloc[-splitPoint:].mean()
        
        trendChange = lastPeriodMean - firstPeriodMean
        trendPct = (trendChange / firstPeriodMean) * 100
        
        print("\nWay 2- Analyze the trend by Comparing first 30% vs last 30% of the period\n")
        print(f"First {splitPoint} days average: {firstPeriodMean:.2f}")
        print(f"Last {splitPoint} days average: {lastPeriodMean:.2f}")
        print(f"Absolute change: {trendChange:+.2f}")
        print(f"Percentage change: {trendPct:+.1f}%")
        
        if trendPct > 5:
            print("Strong upward trend")
        elif trendPct > 1:
            print("Moderate upward trend")
        elif trendPct < -5:
            print("Strong downward trend")
        elif trendPct < -1:
            print("Moderate downward trend")
        else:
            print("Relatively stable trend")
    
    # Way 3- Analyze the trend by Comparing the beginning and the end of the period
    firstHalfMean = df['Sales'].iloc[:len(df)//2].mean()
    secondHalfMean = df['Sales'].iloc[len(df)//2:].mean()
    change_pct = ((secondHalfMean - firstHalfMean) / firstHalfMean) * 100
    print("\nWay 3- Analyze the trend by Comparing the beginning and the end of the period")
    print(f"\nFirst half average: {firstHalfMean:.2f}")
    print(f"Second half average: {secondHalfMean:.2f}")
    print(f"Overall change: {change_pct:+.1f}%")
    
else:
    print("Insufficient data for trend analysis")

# Task 3. Analyzing sales volatility
print("\nTask 3. Sales volatility analysis:\n")

# Calculate variability indicators
sales_std = df['Sales'].std()
sales_mean = df['Sales'].mean()
cv = (sales_std / sales_mean) * 100  # Coefficient of variation

print(f"Sales average: {sales_mean:.2f}")
print(f"Standard deviation: {sales_std:.2f}")
print(f"Coefficient of variation: {cv:.1f}%")

if cv < 20:
    volatility = "Stable"
elif cv < 50:
    volatility = "Moderate"
else:
    volatility = "Highly volatile"

print(f"Volatility level: {volatility}")

# Monthly analysis to understand seasonal  
print("\n" + "-"*50)
print("Monthly analysis:")
df['Month'] = df.index.month
monthly_stats = df.groupby('Month')['Sales'].agg(['mean', 'std'])

print("\nMonthly sales averages:")
for month, stats in monthly_stats.iterrows():
    print(f"  Month {month}' sale: {stats['mean']:.2f} (std: {stats['std']:.2f})")

# Analysis summary
print("\n" + "="*50)
print("Analysis Summary:")
print("="*50)
print(f"- Overall trend: {trend}")
print(f"- Volatility level: {volatility}")
print(f"- Days with significant fluctuations: {len(significantDeviations)}")
print(f"- Maximum sales: {df['Sales'].max()}")
print(f"- Minimum sales: {df['Sales'].min()}")
print(f"- Data period: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
print(f"- Total days analyzed: {len(df)}")
