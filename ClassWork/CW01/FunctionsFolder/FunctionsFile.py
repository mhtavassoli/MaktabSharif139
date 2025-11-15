def Fcn_assign(inputValue):
	if inputValue>0:
		assignment="+"
	elif inputValue<0:
		assignment="-"
	else:
		assignment="0"
	return assignment


"""
def Func_CheckSignOfNumber(number):
	if number>0:
		return "+"
	elif number<0:
		return "-"
	else:
		return "0"
"""

"""
def sign(num):
	if num>0:
		return f"{num} is positive."
	elif num<0:
		return f"{num} is negative."
	else:
		return f"{num} is not positive and negative."
"""

def Fcn_class(input):
	Flag_health=0
	Flag_Fault=0
	Flag_None=0
	if input=="Class1":
		print("Data is correct")
		Flag_health=1
	elif input=="Class2":
		print("Data is incorrect")
		Flag_Fault=1
	else:
		#print("Data is not classified")
		Flag_None=1
		if len(input)<5:
			print(input+" Class1")
		else:
			print(input+" Class2")
	return Flag_health, Flag_Fault, Flag_None
	
"""
def namedetection(name):
	if name=="class1":
		flag_health=1
		return f"name is correct and flag_right is {flag_health} "
	elif name=="class2":
		flag_false=1
		return f"name is incorrect and flag_false is {flag_false} "
	else:
		flag_none=1
		return f"name is not classificated and flag_none is {flag_none}"
"""


def Fcn_Days(input):
	years=input//360
	remaining_days=input%360
	monthes=remaining_days//30
	days=remaining_days%30
	return years, monthes, days
	