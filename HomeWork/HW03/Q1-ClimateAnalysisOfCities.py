import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dayNum = 30
cityNumber = 6
dayArray=np.array(range(dayNum))
cityArray=np.array(["Tehran", "Tabriz", "Isfahan", "Mashhad", "Shiraz", "Qom"])
temperatureArray=np.random.randint(15,40,(cityNumber,dayNum))
humidityArray=np.random.randint(20,80,(cityNumber,dayNum))
rainfallArray=np.random.randint(0,50,(cityNumber,dayNum))
print("\n- part 1: Data generation with NumPy\n")
print(f"The temperature =\n: {temperatureArray}\n")
print(f"The humidity =\n: {humidityArray}\n")
print(f"The rainfall =\n: {rainfallArray}\n")

data=[]
for cityIndex, city in enumerate(cityArray):
    for day in range(dayNum):
        data.append({
            'City': city,
            'Day': day + 1,
            'Temperature': temperatureArray[cityIndex, day],
            'Humidity': humidityArray[cityIndex, day],
            'Rainfall': rainfallArray[cityIndex, day]
        })
df=pd.DataFrame(data)
print(f"\n- part 2: The dataframe\n\n{df}\n")

cityStats = df.groupby('City')[['Temperature', 'Humidity', 'Rainfall']].mean()
print(f"\n- part 3: The Statistical analyses")
print(f"\n- part 3.1: Mean by city:\n\n{cityStats.round(2)}\n")

cityHottest = cityStats['Temperature'].idxmax()
cityColdest = cityStats['Temperature'].idxmin()
print("\n- part 3.2: The city with hottest & coldest mean temperature")
print(f"The city with hottest mean temperature is "
      f"'{cityHottest}' with {cityStats.loc[cityHottest, 'Temperature']:.2f}°C .")
print(f"The city with coldest mean temperature is "
      f"'{cityColdest}' with {cityStats.loc[cityColdest, 'Temperature']:.2f}°C .")

rainyDays = df[df['Rainfall'] > 10].groupby('City').size()
print("\n- part 3.3: The number of days with more than 10 mm of rainfall\n")
print(rainyDays)
print(f"The total number of rainy days is equal to '{rainyDays.sum()}' days .")

print("\n- part 4: Scatter plotting\n")
dfIsfahan = df[df['City'] == 'Isfahan'].drop('City',axis=1)
print(f"dfIsfahan :\n {dfIsfahan}")

"""
plt.figure(figsize=(10, 6))
plt.scatter(dfIsfahan['Temperature'], dfIsfahan['Humidity'], 
           alpha=0.3, color='blue', s=50)  # alpha: Transparency (0 = transparent, 1 = opaque)
plt.title('Temperature and Humidity relationship - Isfahan (30 Days)', fontsize=14)
plt.xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
plt.ylabel('Humidity (%)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
"""

print("\n- part 5: Interpreting and Analyzing scatter plot\n")
IsfahanStats = dfIsfahan.groupby('Temperature').mean()
print(f"\n- part 5.1: Isfahan columns Mean by Temperature:\n\n{IsfahanStats.round(2)}\n")

IsfahanTemperatureHumidityStats = dfIsfahan.groupby('Temperature')['Humidity'].agg([
    ('count', 'count'),
    ('% mean', 'mean'),
    ('% median', 'median'),
    ('% std', 'std'),
    ('% min', 'min'),
    ('% max', 'max')
]).round(2)
print(60*"-","\n Detailed Temperature & Humidity Statistics in Isfahan city:\n")
print(IsfahanTemperatureHumidityStats)



# AQ1. Which temperature has the highest mean humidity?
IsfahanTemperatureHumidityHighestMeanIndex = IsfahanTemperatureHumidityStats['% mean'].idxmax()
IsfahanTemperatureHumidityHighestMeanValue = IsfahanTemperatureHumidityStats['% mean'].max()
IsfahanTemperatureHumidityLowestMeanIndex = IsfahanTemperatureHumidityStats['% mean'].idxmin()
IsfahanTemperatureHumidityLowestMeanValue = IsfahanTemperatureHumidityStats['% mean'].min()
IsfahanTemperatureHumidityHighestMedianIndex = IsfahanTemperatureHumidityStats['% median'].idxmax()
IsfahanTemperatureHumidityHighestMedianValue = IsfahanTemperatureHumidityStats['% median'].max()
print("\n",60*"-")
print(f"   • Temperature of Isfahan with highest Mean Humidity: "
      f"{IsfahanTemperatureHumidityHighestMeanIndex}°C (%{IsfahanTemperatureHumidityHighestMeanValue})")
print(f"   • Temperature of Isfahan with lowest Mean Humidity: "
      f"{IsfahanTemperatureHumidityLowestMeanIndex}°C (%{IsfahanTemperatureHumidityLowestMeanValue})")
print(f"   • Temperature of Isfahan with highest Median Humidity: "
      f"{IsfahanTemperatureHumidityHighestMedianIndex}°C (%{IsfahanTemperatureHumidityHighestMedianValue})")
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

for temperature in dfIsfahan['Temperature'].unique():
    temperatureHumidity = dfIsfahan[dfIsfahan['Temperature'] == temperature]['Humidity']
    outliers = detect_outliers(temperatureHumidity)
    print(f"   • {temperature}°C: {len(outliers)} outliers (Humidity: %{list(outliers.round(2))})")


# AQ3. Pattern analysis between Cold, Moderate and Hot temperatures of Isfahan

print(f"\n3. Cold, Moderate and Hot temperatures Isfahan PATTERNS:")
# Define Cold, Moderate and Hot temperature
coolTemperature = [i for i in range(15, 24)]
moderateTemperature = [i for i in range(24, 27)]
hotTemperature = [i for i in range(27, 40)]

coolTemperatureHumidityIsfahan = dfIsfahan[dfIsfahan['Temperature'].isin(coolTemperature)]['Humidity']
moderateTemperatureHumidityIsfahan = dfIsfahan[dfIsfahan['Temperature'].isin(moderateTemperature)]['Humidity']
hotTemperatureHumidityIsfahan = dfIsfahan[dfIsfahan['Temperature'].isin(hotTemperature)]['Humidity']


print("\n")
print(f"   • Cool temperature ({', '.join([str(i) for i in coolTemperature])})°C:") 
                            # type(', '.join([str(i) for i in coolTemperature])) : str
print(f"     - Average humidity: %{coolTemperatureHumidityIsfahan.mean():.2f}")
print(f"     - Median humidity: %{coolTemperatureHumidityIsfahan.median():.2f}")
print(f"     - Number of measurements: {len(coolTemperatureHumidityIsfahan)}")
print("\n")
print(f"   • Moderate temperature ({', '.join(map(str, moderateTemperature))})°C:") 
                            # type(', '.join(map(str, moderateTemperature))) : str
print(f"     - Average humidity: %{moderateTemperatureHumidityIsfahan.mean():.2f}")
print(f"     - Median humidity: %{moderateTemperatureHumidityIsfahan.median():.2f}")
print(f"     - Number of measurements: {len(moderateTemperatureHumidityIsfahan)}")
print("\n")
print(f"   • Hot temperature ({(*hotTemperature,)})°C:")                            
                            # type(*hotTemperature,) : 'tuple'
print(f"     - Average humidity: %{hotTemperatureHumidityIsfahan.mean():.2f}")
print(f"     - Median humidity: %{hotTemperatureHumidityIsfahan.median():.2f}")
print(f"     - Number of measurements: {len(hotTemperatureHumidityIsfahan)}")


# Statistical comparison
difference = coolTemperatureHumidityIsfahan.mean() - moderateTemperatureHumidityIsfahan.mean()
if coolTemperatureHumidityIsfahan.mean() > moderateTemperatureHumidityIsfahan.mean():
    print(f"   Average coolTemperatureHumidityIsfahan are %{difference:.2f} HIGHER on average than moderateTemperatureHumidityIsfahan")
else:
    print(f"   Average moderateTemperatureHumidityIsfahan are %{-difference:.2f} HIGHER on average than coolTemperatureHumidityIsfahan")

difference = coolTemperatureHumidityIsfahan.mean() - hotTemperatureHumidityIsfahan.mean()
if coolTemperatureHumidityIsfahan.mean() > hotTemperatureHumidityIsfahan.mean():
    print(f"   Average coolTemperatureHumidityIsfahan are %{difference:.2f} HIGHER on average than hotTemperatureHumidityIsfahan")
else:
    print(f"   Average hotTemperatureHumidityIsfahan are %{-difference:.2f} HIGHER on average than coolTemperatureHumidityIsfahan")

difference = hotTemperatureHumidityIsfahan.mean() - moderateTemperatureHumidityIsfahan.mean()
if hotTemperatureHumidityIsfahan.mean() > moderateTemperatureHumidityIsfahan.mean():
    print(f"   Average hotTemperatureHumidityIsfahan are %{difference:.2f} HIGHER on average than moderateTemperatureHumidityIsfahan")
else:
    print(f"   Average moderateTemperatureHumidityIsfahan are %{-difference:.2f} HIGHER on average than hotTemperatureHumidityIsfahan")

# Additional insights
print(f"\n4. ADDITIONAL INSIGHTS:")
print(f"   • Total Temperature number: {len(dfIsfahan['Temperature'].unique())}")
print(f"   • Total Temperature smaples: {len(dfIsfahan)}")
print(f"   • Overall average Humidity: %{dfIsfahan['Humidity'].mean():.2f}")
print(f"   • Humidity amount range: %{dfIsfahan['Humidity'].min():.2f} - %{df['Humidity'].max():.2f}")

# Temperature with most consistent Humidity (lowest standard deviation)
mostConsistentHumidity = IsfahanTemperatureHumidityStats['% std'].idxmin()
leastConsistentHumidity = IsfahanTemperatureHumidityStats['% std'].idxmax()
print(f"   • Most consistent Humidity: {mostConsistentHumidity}°C (std: %{IsfahanTemperatureHumidityStats.loc[mostConsistentHumidity, '% std']:.2f})")
print(f"   • Least consistent Humidity: {leastConsistentHumidity}°C (std: %{IsfahanTemperatureHumidityStats.loc[leastConsistentHumidity, '% std']:.2f})")
