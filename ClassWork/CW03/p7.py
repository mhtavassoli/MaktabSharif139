import numpy as np
import pandas as pd
priceArray=np.random.randint(1000,9000,8)
areaArray=np.random.randint(60,300,8)
cityArray=np.array(["Tehran", "Tabriz", "Isfahan", "Mashhad", "Shiraz", "Qom", "Kerman", "Yazd"])
df=pd.DataFrame({'City':cityArray,'Area':areaArray,'Price':priceArray})

print("-"*80)
print("\n 4- DataFrame is: \n\n",df)
df['Price/Area']=df['Price']/df['Area']
print("\n 5- DataFrame with 'Price/Area' is: \n\n",df)

averagePriceCity=df['Price/Area'].mean()
print("\n 6- Average of 'Price/Area' of country =\n\n",averagePriceCity)

dfPriceUpAveragePriceCity=df[df['Price/Area']>averagePriceCity]
print("\n 7.1- DataFrame with condition 'Price > average(country's Price/Area)' is: \n\n",dfPriceUpAveragePriceCity)

cityPriceUpAveragePriceCity=dfPriceUpAveragePriceCity.loc[:,['City','Price/Area']]
print("\n 7.2- Cities with condition 'Price > average(country's Price/Area)' is: \n\n",cityPriceUpAveragePriceCity)

dfSortedByPrice=df.sort_values(by='Price',ascending=False)
print("\n 8- DataFrame sorted by 'Price' is: \n\n",
      dfSortedByPrice.loc[:,df.columns.difference(['Price/Area'])])                    # loc, list of droped columns
# df.sort_values(by='Price',ascending=False,inplace=True)                              # inplace, update df
# print("\n DataFrame sorted by 'Price' is: \n\n",df.loc[:,['City','Area','Price']])   # loc, list of columns
