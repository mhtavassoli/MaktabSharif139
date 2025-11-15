def clean_string(strInput):
    strLower = strInput.lower()
    strRemoveSpace = strInput.replace(" ", "")
    numbers = ""
    for c in strInput:
        if c.isdigit():
            numbers += c
    if numbers == "":
        numbers = "There is no number in {strInput}"
    return strLower, strRemoveSpace, numbers
####################################################################################################
strInput1 = input("Please enter the Strin: ")
Result = clean_string(strInput1)
print(f"strLower is {Result[0]}, strRemoveSpace is {Result[1]}, numbers is {Result[2]}")
print("#"*100)