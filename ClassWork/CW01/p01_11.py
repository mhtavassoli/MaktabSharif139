a=int(input("please enter the number1: "))
# b=int(input("please enter the number2: "))
b=0
# operator=input("please choose +,-,* or / ")
operator="/"

if operator=="+":
	print(a+b)
elif operator=="-":
	print(a-b)
elif operator=="*":
	print(a*b)
elif operator=="/":
	if(b==0):
		print("The denominator is zero")
	else:
		print(a / b)    
else:
    print("The operator is not defined")
