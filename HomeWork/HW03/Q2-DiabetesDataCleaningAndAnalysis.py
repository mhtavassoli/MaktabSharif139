import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1- Read the CSV file and explore the data
df=pd.read_csv(r"E:\Maktab\Artificial Intelligence\Programming\HomeWork\HW03\diabetes.csv")
print("\n",df,"\n","-"*50,"1- DataFrame")
df.info()

dfFilled=df.copy()  # make a copy
dfFilled = df.fillna(df.mean())
# dfFilled.fillna(dfFilled.mean(numeric_only=True), inplace=True)

print(f"\n- part 1.1: \n\n {dfFilled}")
dfFilledMissing = dfFilled.isnull().sum()
print(f"\n- part 1.2: way 1 \n\n {dfFilledMissing}")
print(f"\n way 2 to show null value: {dfFilled.isnull().any().any()}")
print(f"\n part 1.3: {dfFilled.head()}")

Columns6=['Glucose', 'BloodPressure', 'BMI', 'Age', 'Insulin', 'Outcome']
correlationMatrix = dfFilled[Columns6].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlationMatrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.2f',
            linewidths=0.5)
plt.title('Correlation matrix of 6 columns')
plt.show()

maxCorrFeature=correlationMatrix['Outcome'].drop('Outcome').idxmax()
maxCorrValue=correlationMatrix['Outcome'].drop('Outcome').max()
print(f"\n maxCorrFeature is {maxCorrFeature} with value of {maxCorrValue}.")
