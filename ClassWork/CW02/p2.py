def detect_equal(str1,str2):
	isTypeEqual=False
	isValueEqual=False
	if(str1.isdigit() and str2.isdigit()):
		if(type(str1)==type(str2)):
			isTypeEqual=True
		if(int(str1)==int(str2)):
			isValueEqual=True
	return isTypeEqual,isValueEqual

str1=input("Enter Your num1:")
str2=input("Enter Your num2:")
equalType,equalValue=detect_equal(str1,str2)
if(equalType==True):
	print("Type Is Equal")
else:
	print("Type IS Not Equal")
if(equalValue==True):
	print("Value Is Equal")
else:
	print("Value IS Not Equal")
	
	
