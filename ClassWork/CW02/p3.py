def Fcn_convert(strInput):
    try:
        floatNumber=float(strInput)
        print(f"data type of {strInput} is : {type(floatNumber)} and data value: {floatNumber}")
    except ValueError:
        print("Error in converting input to float")
    try:
        intNumber=int(strInput)
        print(f"data type of {strInput} is: {type(intNumber)} and data value: {intNumber}")
    except ValueError:
        print("Error in converting input to int")
    try:
        strNumber=str(strInput)
        print(f"data type of {strInput} is: {type(strNumber)} and data value: {strNumber}")
    except ValueError:
        print("Error in converting input to str")     
    print("_"*50)   
    
for i in [12, 12.0, "ali"]:
    Fcn_convert(i)
    
