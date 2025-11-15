def Fcn_order2(num1,num2):
    if num1>num2:
        return num1
    else:
        return num2
    
def Fcn_order3(num1,num2,num3):
    if num1>num2 and num1>num3:
        return num1
    elif num2>num1 and num2>num3:
        return num2
    else:
        return num3
    return num3

def Fcn_orderN(number):
    print(f"You want to enter {number} numbers.")
    result=int(input("Please enetr the first number: "))
    for i in range(number-1):
        num=int(input("Please enetr the next number: "))
        if result<num:
            result=num
    print("The last number was also received from the input.")
    return result

        
num1=12
num2=15
num3=14
numberN=10 # input N number to find max in those.
print(f"The max number is: {Fcn_order2(num1,num2)}")
print(f"The max number is: {Fcn_order3(num1,num2,num3)}")
print(f"The max number is: {Fcn_orderN(numberN)}")