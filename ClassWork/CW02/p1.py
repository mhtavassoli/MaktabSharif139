def detect_type(inputValue):
	if inputValue in [True , False]:
		return "Bool"
	try:
		if int(inputValue)==inputValue:
			return "int"
	except:
		pass	
	try:	
		if float(inputValue)==inputValue:
			return "float"
	except:
		pass
	try:
		if type(inputValue).__name__=="str":
			return "str"
	except:
		pass
	else:
		return "unknown"

inputVector=[12, 12.3, True, "ali", 1+2j]
for i in inputVector:
    print(f"The data type of input is: {detect_type(i)}")
