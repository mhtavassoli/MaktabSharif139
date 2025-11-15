def analysisFunction(str):
	cLowerContant=0
	CUpperContant=0
	for c in str:
		if c.lower()!=c:
			cLowerContant+=1
		if c.upper()!=c:
			CUpperContant+=1
	CRemovedSpaceContant=len(str)-len(str.replace(" ",""))
	Result={
			"cLowerContant": cLowerContant,
			"CUpperContant": CUpperContant,
			"CRemovedSpaceContant": CRemovedSpaceContant}
	return Result
####################################################################################################
strInput=input("Please enter the String: ")
print(analysisFunction(strInput))
