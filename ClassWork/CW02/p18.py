def UpperLowerCounter(str):
    UpperCounter=0
    LowerCounter=0
    for i in str:
        if not i.isdigit():
            if i.upper() ==i:
                UpperCounter+=1
            elif i.lower()==i:
                LowerCounter+=1
    return LowerCounter,UpperCounter

inputString="Ali8Zahra"
inputStringLowerCounter, inputStringUpperCounter=UpperLowerCounter(inputString)
print(f"The {inputString} with length of {len(inputString)} characters has "
      f"{inputStringLowerCounter} lower characters and {inputStringUpperCounter} upper characters")