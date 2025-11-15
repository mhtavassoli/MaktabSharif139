number_1= float(input ("please enter your number 1 :"))
number_2=float (input("please enter your number 2 :"))
operator= input("please enter operator ")
if operator == "*":
	print(f"your answer is {number_1*number_2}")
elif operator=="+":
	print(f"your answer is {number_1+number_2}")
elif operator=="-":
	print(f"your answer is {number_1-number_2}")
elif operator=="/" :
	try:
		res = number_1 / number_2
		print(res)
	except ZeroDivisionError:
		print("0")
else:
	print("your operator is not defined")
 
 
 
"""
a=int(input("please enter the number1: "))
b=int(input("please enter the number2: "))
operator=input("please choose +,-,* or / ")
if operator=="+":
	print(a+b)
elif operator=="-":
	print(a-b)
elif operator=="*":
	print(a*b)
elif operator=="/":
    try:
        res = a / b
    except ZeroDivisionError:     # except (ZeroDivisionError, ValueError):
        res = 0                   # print('Please enter a number another than zero')
    print(res)
"""