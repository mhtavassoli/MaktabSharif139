def SumOddInterval(number1, number2):
    if number1>number2 or (number1==number2 and number1%2==0):
        return "None"
    elif number1%2==1: 
        Result=number1
    else:
        Result=number1+1
    for i in range(number1+2,number2+1,2):
        Result+=i
    return Result

num1=int(input("Please enter the first number: "))
num2=int(input("Please enter the second number: "))
print(f"The Sum of odd numbers between {num1} and {num2} is equal to {SumOddInterval(num1,num2)}")
