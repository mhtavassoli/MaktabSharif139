number1=int(input('Please enter the first Number: '))
number2=int(input('Please enter the second Number: '))
operator=input('Please enter the operator: ')
if operator == "+":
	result = number1 + number2
elif operator == "-":
	result = number1 - number2
elif operator == "*":
	result = number1 * number2
elif operator == "/":
	while number2==0:
		number2=int(input('Please enter the non-zero Number: '))
	result = number1 / number2
			
print(f"the result of {number1} {operator} {number2} is {result}")
