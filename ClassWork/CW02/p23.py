
def reversePrintFunction(inputNumber):
	for i in range(inputNumber,0,-1):
		for j in range(i, 0, -1):
			print(j ,end=" ")
		print()
	
reversePrintFunction(5)


"""
def print_num(num):
	step=num
	while(step>0):
		for i in range(step,0,-1):
			print(i,end=" ")
		print("\n")
		step-=1
print_num(5)
"""

"""
def print_num(num):
	for i in range(num,0,-1):
		for j in range(i,0,-1):
			print(j, end=' ')
		print()
print_num(5)
"""
