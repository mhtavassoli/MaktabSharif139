def SumPositiveNumbers():
    Result=0
    while True:
        number=int(input("Please enter the positive number to sum or negative number to end: "))
        if number>0:
            Result+=number
        else:
            break
    return Result

print(f"The sum of input numbers is equal to {SumPositiveNumbers()} .")