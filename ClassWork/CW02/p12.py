def RepetitionInListFunction(listInput):
	seen=[]
	duplicates=[]
	for item in listInput:
		if item in seen:
			duplicates.append(item)
		else:
			seen.append(item)    
	return duplicates
listInput=[1, 2, 3, 2, 4, 3, 5]
print("Duplicated Numbers are: ", RepetitionInListFunction(listInput))

"""
from itertools import count
def detect_inputList(inuputList):
	repeatedSet=set()
	for i in inuputList:
		if(inuputList.count(i)>1):
			repeatedSet.add(i)
	return repeatedSet

result=detect_inputList([1,1,1,45,67,54,67])
print(result)
"""

"""
inuputList=[1,1,1,45,67,54,67]
repeatedSet=set()
for i in range(len(inuputList)):
	for j in range(i + 1, len(inuputList)):
		if inuputList[i] == inuputList[j]:
			repeatedSet.add(inuputList[i])
print(repeatedSet)
"""
