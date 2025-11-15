def findMaxMinFunction(listInput):
	numberMax=listInput[0]
	numberMin=listInput[0]
	for i in listInput:
		if numberMax<i:
			numberMax=i
		if numberMin>i:
			numberMin=i
	return numberMin, numberMax

listInput=[3, 2, 5, 6, 4]
print(f"The numberMin and numberMax are {findMaxMinFunction(listInput)}")

		