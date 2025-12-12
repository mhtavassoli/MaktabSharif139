import pandas as pd
import numpy as np
# from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r"E:\Maktab\Artificial Intelligence\Programming\ClassWork\CW4\heart.csv")
print("\n",df,"\n","-"*50,"1- DataFrame")

numericColumns = df.select_dtypes(include=[np.number]).columns
# way2 :
# col_numeric=[]
# for i in df.columns:
#     if "int" in str(df[i].dtypes) or "float" in str(df[i].dtypes):
#         col_numeric.append(i)       
print(f"The number of numerical columns:\n{len(numericColumns)} \n")
print(f"The numerical columns: {list(numericColumns)}\n ")

dfNumeric = df[numericColumns] 
print(f"\nDataFrame included numerical columns: \n\n {dfNumeric}")

correlationMatrix = dfNumeric.corr()
print(f"\nThe correlation Matrix: \n\n {correlationMatrix}\n ")

print("-"*80,"\n Correlation Matrix Analysis:")
targetCorrelations = correlationMatrix['target'].drop('target').sort_values(ascending=False) # Correlations with target
print("\n Correlation of all features with target (descending order):","-" * 60)
for feature, corr in targetCorrelations.items():
    direction = "Positive" if corr > 0 else "Negative"
    strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
    print(f"{feature:15}: {corr:7.3f} | {direction} | {strength}")
print("\n" + "="*50)

mostNegativeFeature = targetCorrelations.idxmin()
mostNegativeValue = targetCorrelations.min()
print(f"\n Feature with the most negative correlation with target:")
print(f"-> Feature: {mostNegativeFeature}")
print(f"-> Correlation value: {mostNegativeValue:.3f}")

plt.figure(figsize=(10, 6))
sns.heatmap(correlationMatrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.2f',
            linewidths=0.5)
# data: A 2D dataset that can be coerced into a NumPy ndarray.
# vmin, vmax: Values to anchor the colormap; if not specified, they are inferred from the data.
# cmap: The colormap for mapping data values to colors.
# center: Value at which to center the colormap when plotting divergent data.
# annot: If True, displays numerical values inside the cells.
# fmt: String format for annotations.
# linewidths: Width of the lines separating cells.
# linecolor: Color of the separating lines.
# cbar: Whether to display a color bar.

plt.title('Correlation matrix of Heart Disease Dataset')
plt.tight_layout()          # provides a simple solution to automatically adjust subplot parameters and 
                            # ensure that the plots are neatly spaced with no overlap.
plt.show()
