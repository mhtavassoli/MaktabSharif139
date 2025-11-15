Repeat = "Continue"
while Repeat!="exit":
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
        if b == 0:
            print("Denominator is zero")
        else:
            print(a / b)
    Repeat=input("Do you not want to calculate again, Please type 'exit': ")
print("Calculation end!")
    
    
"""  
def switch_case(operator,num1,num2):
	match operator:
		case "+":
			return f"{num1+num2}"
		case "-":
			return f"{num1-num2}"
		case "*":
			return f"{num1*num2}"
		case "/":
			if(num2==0):
				return "Enter Valid Number"
			else:
				return f"{num1/num2}"
		case _:
			return "Enter Valid Operator"	

c=True
while(c):
	print("Enter Your Operator : ")
	operator=input()
	print("Enter Your First Number : ")
	num1=float(input())
	print("Your Your Second Number :: ")
	num2=float(input())
	result = switch_case(operator, num1, num2)
	print(result)
	print("Do You Want To Continue : ")
	response=input()
	if response=="no":
		c=False
		break	
"""
    
"""
while(True):
    number_1= float(input ("please enter your number 1 :"))
    number_2=float (input("please enter your number 2 :"))
    operator= input("please enter operator ")
    if operator == "*":
        print(f"your answer is {number_1*number_2}")
        break
    elif operator=="+":
        print(f"your answer is {number_1+number_2}")
        break
    elif operator=="-":
        print(f"your answer is {number_1-number_2}")
        break
    elif operator=="/":
        print(f"your answer is {number_1/number_2}")
        break
    else:
        print("your character is not defined")
print("welldone : )")
"""