def evenSumFunction(number):
	if number<0:
		Result="Impossible because of the negative input number"
	else:
		Result=0
		for i in range(number+1):
			if i%2==0:
				Result+=i
	return Result

inputNumber=int(input("Please enter the number: "))
print(f"The Sum of even numbers from 0 to {inputNumber} is: {evenSumFunction(inputNumber)}")


"""
def sumofevennumbers(num):

	if num>0:
		count=0
		for i in range(0,num+1):
			if i%2==0:
				count=count+i
		return count
	else:
		return 'not found'
"""		
		
"""		
def sum_evennum(num):
	if(num<0):
		return "Number Is Negetiv"
	s=0
	for i in range(0,num+1):
		if(i%2==0):
			s+=i
	return s
print(sum_evennum(-9))
"""

"""
num=int(input("Please enter nember:" ))
if num<0:
	print("Please enter even number.")
else:
	sum=0
for i in range(0, num +1, 2):
	sum+=i
print(f"sum {num} is : {sum}")
"""