def analyze_input(inputValue):
    try:
        if type(inputValue)==str:
            print(f"The input '{inputValue}' is string with length of {len(inputValue)} "
                f"and having number situation is {any(char.isdigit() for char in inputValue)}.")
        elif str(inputValue).isdigit() or str(inputValue).replace(".","").isdigit():
            if type(inputValue)==int:
                print(f"The input '{inputValue}' is int.")
            elif type(inputValue)==float:
                print(f"The input '{inputValue}' is float.")
        else:
            print(f"The input '{inputValue}' has unknown data type.")
    except:
        pass
            
inputDictionary = {
  "NumberInt": 12,
  "NumberFloat": 3.4,
  "String": "str1",
  "Bool": True
}
for i in inputDictionary:
    analyze_input(inputDictionary[i])
        
