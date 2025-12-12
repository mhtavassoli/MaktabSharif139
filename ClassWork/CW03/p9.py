from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv(r"E:\Maktab\Artificial Intelligence\Programming\ClassWork\CW4\icecream_data.csv")
print("-"*80,"DataFrame is:\n",df)                # Pandas will only return the first 5 rows, and the last 5 rows
print("-"*80,"DataFrame is:\n",df.to_string())    # This method returns the entire DataFrame

# This method returns the headers and a specified number of rows, starting from the top.
print("-"*80,"DataFrame head is:\n",df.head()) # This method is for viewing the first rows of the DataFrame

# This method returns the headers and a specified number of rows, starting from the bottom.
print("-"*80,"DataFrame tail is:\n",df.tail()) # This method is for viewing the last rows of the DataFrame

print("-"*80,"DataFrame info is:\n")
print(df.info()) # gives you more information about the data set.

plt.scatter(df['Temperature'], df['IceCreamSold'],color='red',marker='*')
plt.xlabel('Temperature (°C)',fontweight='bold')
plt.ylabel('Ice Cream Sold (kg)',fontweight='bold')
plt.title('Temperature vs Ice Cream Consumption')
plt.show()
