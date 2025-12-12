def detect_NumbersInList(inuputString):
	numberList=[]
	mainstring=inuputString.split(",")
	for item in mainstring:
		try:
			if(item.isdigit()):
				numberList.append(int(item))
		except:
			pass
	return numberList

#inputString=input("Enter Input String:")
inputString="40,abc,10,20,30"
result=detect_NumbersInList(inputString)
print(result)

