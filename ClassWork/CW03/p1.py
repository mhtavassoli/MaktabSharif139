import pandas as pd
books=pd.DataFrame({
    "title": ["Harry Potter", "Anne of Green Gables", "The birth of a killer"],
    "price": [98000, 76000, 89000],
    "auther": ["J.K. Rowling", "L.M. Montgomery", "Darren Shan"]
},index=["b1","b2","b3"])
print(books)
print(50*"-")
subset1_iloc=books.iloc[0:2,0:2]
print(subset1_iloc)
print(50*"-")
subset1_loc=books.loc["b1":"b2",["title","price"]]
print(subset1_loc)
print(50*"-")
try:
    subset2=books.loc[1:2,["title","price"]]
    print(50*"-")
except :
    print(f'Error!: "subset2=books.loc[1:2,["title","price"]]" '
          f'\nCorrect:"subset2=books.iloc[1:2,0:2]"'
          f'\n{50*"-"}')
    subset2=books.iloc[1:2,0:2]
    print(subset2)
    print(50*"-")

"""
books0=pd.DataFrame({
    "title": ["Harry Potter", "Anne of Green Gables", "The birth of a killer"],
    "price": [98000, 76000, 89000],
    "auther": ["J.K. Rowling", "L.M. Montgomery", "Darren Shan"]
},index=[0,1,2])

print("-----------books--------------")
print(books0)

print("--------- Subset 0------------")
subset0=books0.iloc[0:2,0:2]
print(subset0)

print("--------- Subset 1 ------------")
books1=pd.DataFrame({
    "title": ["Harry Potter", "Anne of Green Gables", "The birth of a killer"],
    "price": [98000, 76000, 89000],
    "auther": ["J.K. Rowling", "L.M. Montgomery", "Darren Shan"]
},index=["b1","b2","b3"])

subset1=books1.loc["b1":"b2",["title","price"]]
print(subset1)

print("----------subset 2 -----------")

books2=pd.DataFrame({
    "title": ["Harry Potter", "Anne of Green Gables", "The birth of a killer"],
    "price": [98000, 76000, 89000],
    "auther": ["J.K. Rowling", "L.M. Montgomery", "Darren Shan"]
},index=["b1","b2","b3"])

subset2=books2.loc[1:2,["title","price"]]
print(subset2)

print("----------subset 3 -----------")

books3=pd.DataFrame({
    "title": ["Harry Potter", "Anne of Green Gables", "The birth of a killer"],
    "price": [98000, 76000, 89000],
    "auther": ["J.K. Rowling", "L.M. Montgomery", "Darren Shan"]
})

subset3=books3.loc["b1":"b2",["title","price"]]
print(subset3)
"""
