def powerFunction(base,power):
    result=1
    if type(base)==int:
        while power>0:
            result*=base
            power-=1
    return result
baseValue=int(input("Please enter the base: "))
powerValue=int(input("Please enter the power: "))
print(f"base**power or {baseValue} ^ {powerValue} is equal to {powerFunction(baseValue,powerValue)}")
    