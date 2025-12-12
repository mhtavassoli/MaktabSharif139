import pandas as pd
# data = {
# "shop_names": ["Ali", "Sara", "Reza"],
# "city_names": ["shiraz", "tehran", "esfahan"],
# "categories": ["Food", "Product", "Grocory"]
# }

# myseries1 = pd.Series(['Ali','Sara','Reza'],index=['a','b','c'])
myseries1 = pd.Series(['Store1','Store2','Store3'])
myseries2 = pd.Series(['Shiraz','Tehran','Esfahan'])
myseries3 = pd.Series(['Clothing','Grocery','Stationery'])

df=pd.DataFrame({"shop_names": myseries1,
                 "cities"    : myseries2,
                 "categories": myseries3})
"""
df=pd.DataFrame({"shop_names": myseries1,
                 "cities": myseries2,
                 "categories": myseries3},
                 index=list("ABC"))
"""
print(df,"\n",50*"-","*1")

df.index=["A", "B", "C"]
print(df,"\n",50*"-","*2")
print(df["shop_names"],"\n",50*"-","*3-1")

print(df["shop_names"].values,"\n",50*"-","*3-1 better 1")
print(df["shop_names"].to_string(index=False),"\n",50*"-","*3-1 better 2")
print(df["categories"],"\n",50*"-","*3-2-1")
print(df["categories"].describe(),"\n",50*"-","*3-2-2")
print(df.sort_values(by=["categories"],ascending=True),"\n",50*"-","*3-3")


"""
import pandas as pd
# data = {
# "shop_names": ["Ali", "Sara", "Reza"],
# "city_names": ["shiraz", "tehran", "esfahan"],
# "categories": ["Food", "Product", "Grocory"]
# }
# data1 = {'shop_names': 'Ali', 'city_names': 'shiraz', 'categories': 'Food'}
# data2 = {'shop_names': 'Ali', 'city_names': 'shiraz', 'categories': 'Food'}
# data3 = {'shop_names': 'Ali', 'city_names': 'shiraz', 'categories': 'Food'}
# myseriescolumns = pd.Series(['shop_names','city_names','categories'],index=['a','b','c'])
myseries1 = pd.Series(['Ali','Sara','Reza'],index=['a','b','c'])
myseries2 = pd.Series(['shiraz','tehran','esfahan'],index=['a','b','c'])
myseries3 = pd.Series(['Food','Product','Grocory'],index=['a','b','c'])

dataframe1=pd.DataFrame({"shop_names":myseries1,"cities":myseries2,"categories":myseries3})
print("-------1 . --------------")
print(dataframe1)
print("-------2 . --------------")
# dataframe2=pd.DataFrame(,index=["a","b","c"])
print(dataframe1["shop_names"])
print("-------3 . --------------")
print(dataframe1["categories"])
print("------- 4 . --------------")
print(dataframe1["categories"].describe())
print("------- 5 . --------------")
print(dataframe1.sort_values(by=["categories"],ascending=True))
"""