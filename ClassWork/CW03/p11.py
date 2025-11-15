import pandas as pd
df=pd.read_csv(r"E:\Maktab\Artificial Intelligence\Programming\ClassWork\CW4\students_scores.csv")
print("\n",df,"\n","-"*50,"0- DataFrame")

dfFilled=df.copy()  # make a copy

# for column in df.columns[df.isnull().sum>0]:
#     df[column]=df[column].fillna(df[column].mean)

# df.fillna(df.mean(numeric_only=True), inplace=True)

means = df[['math', 'science', 'english']].mean()    # Calculate mean of each column (ignoring NaN values)
dfFilled[['math', 'science', 'english']] = dfFilled[['math', 'science', 'english']].fillna(means) # Replace missing values with column means
print("\n",dfFilled,"\n","-"*50,"1- DataFrame Filled with column means")

dfFilled=df.copy()  # make a copy
means=dfFilled.drop('name', axis=1).mean()
dfFilled[means.index] = dfFilled[means.index].fillna(means)
print("\n",dfFilled,"\n","-"*50,"2- DataFrame Filled with column means")

dfFilled=df.copy()  # make a copy
means=dfFilled.loc[:, df.columns != 'name'].mean()
dfFilled.update(dfFilled[means.index].fillna(means))
print("\n",dfFilled,"\n","-"*50,"3- DataFrame Filled with column means")

dfDropped=df.dropna()      # Remove rows containing NaN values
print("\n",dfDropped,"\n","-"*50,"4- DataFrame Dropped rows containing NaN values")
