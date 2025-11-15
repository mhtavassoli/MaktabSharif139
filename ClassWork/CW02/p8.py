def CombinedList(listInput):
    listNumbers=[]
    listStr=[]
    for i in listInput:
        if i ==str(i):
            listStr.append(i)
        elif int(i)==i or float(i)==i:
            listNumbers.append(i)
        else:
            print(f"The data type of this item {i} is not int, float or string!")
            
    return listNumbers, listStr
        
listInput1=[2, 2.3, "Ali", 4, 5.6, "Zahra" ]
print(CombinedList(listInput1))
