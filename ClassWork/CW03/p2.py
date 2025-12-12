import pandas as pd
data= {
"name": ["Ali", "Sara", "Reza", "Fatemeh"],
"age": [21, 22, 20, 23],
"major": ["AI", "Data Science", "Network", "AI"]
}
df=pd.DataFrame(data)
print(df,"\n",50*"-","*1")
df=df.rename(columns={"major":"field_of_study"})
print(df,"\n",50*"-","*2")
df=df.sort_values(by="age",ascending=True)
print(df,"\n",50*"-","*3")
dataAI=df[df["field_of_study"]=="AI"]
print(dataAI,"\n",50*"-","*4")
print(dataAI.loc[:,["name","age"]],"\n",50*"-","*4 better")
