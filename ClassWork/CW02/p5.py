def sumFunction (numbers):
	Result=0
	for num in numbers:
		Result+=num
	return Result
	
def NegativeNumbersCount(numbers):
	counetr=0
	for num in numbers:
		if num<0:
			counetr+=1
	return counetr
	
def IsNegativeNumber(numbers):
    Result=False
    for num in numbers:
        if num<0:
                Result=True
                break
    return Result

def inputFunction():
	listNumber=int(input("Please enter the numbers of list: "))
	numbers=[]
	for i in range(listNumber):
		numberNew=int(input("Please enter the new number: "))
		numbers=[numbers ,numberNew ]
	return numbers

def compareFunction(numbers):
	if NegativeNumbersCount(numbers)>0 and sumFunction(numbers)>50:
    # if IsNegativeNumber(numbers) and sumFunction(numbers)>50:
		print("the Conditions is met")
	else:
		print("the Conditions is not met")
	return
####################################################################################################	
numbersInput=inputFunction()
compareFunction(numbersInput)
#compareFunction([1, 2, 3])
	
	
"""
def conditions(num1,num2,num3):
    if (num1<0 or num2<0 or num3<0) and num1+num2+num3>50:
        return "condition is True"
    else:
        return "condition is False"
"""

"""
import re       # RegexPython, Regular Expressions
def clean_string(inputValue):
    inputValue=str.lower(inputValue)
    inputValue=inputValue.strip()
    print("Lower Input And Strip Is : ",inputValue)
    numbers = re.findall(r'\d+', inputValue)
    texts=re.findall(r'\D+', inputValue)
    print("Text Is : ",texts)
    if not numbers:
        print("There Is No Number In Input String")
    else:
        print(numbers)

inputValue=input("Enter Something Please:")
clean_string(inputValue)
"""
	