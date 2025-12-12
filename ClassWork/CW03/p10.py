import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv(r"E:\Maktab\Artificial Intelligence\Programming\ClassWork\CW4\monthly_sales.csv")
print("\n",df,"\n","-"*50,"DataFrame")

print("\n",df.info(),"\n","-"*50,"DataFrame info") # It's good for clearing data like null data

dfMean=df.groupby(['month','product'])['sales'].mean() # Calculate average sales for each group by month and product
print("\n",dfMean,"\n","-"*50,"DataFrame by grouping , mean")

dfMean2=df.groupby(['month','product'])['sales'].mean().reset_index()
print("\n",dfMean2,"\n","-"*50,"DataFrame by grouping , mean and reset index ")

dfMean3=df.groupby(['month','product'])['sales'].mean().reset_index(name='MeanSales')
print("\n",dfMean3,"\n","-"*50,"DataFrame by grouping , mean and reset index and naming Column")

df['MeanSalesColumn'] = df.groupby(['month','product'])['sales'].transform('mean')
print("\n",df,"\n","-"*50,"DataFrame by grouping , mean and reset index and adding named Column ")

# converts data from multi-index format to a regular table, Each column will represent a product
dfMeanUnstack=dfMean.unstack() 
print("\n",dfMeanUnstack,"\n","-"*50,"DataFrame by grouping , mean in table mode")   

dfMeanUnstack.plot(kind='bar', width=0.8, figsize=(11,6), color=['blue', 'orange', 'purple'])
# colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown'], default3: ['blue', 'orange', 'green']
plt.title("Average Monthly Sales by Product", fontsize=14)
plt.xlabel('Month', fontweight='bold')
plt.ylabel('Average Sales', fontweight='bold')
plt.legend(title='Product',bbox_to_anchor=(1.13, 1), loc='upper right')
plt.show()
