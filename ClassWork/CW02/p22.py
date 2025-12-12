def startriangle(n):
    if n%2==0:
        print("Please enter the odd number!")
    else:
        for i in range(n):
            # spaces (left)
            for j in range(n - i - 1):
                print(" ", end="")
            # stars
            for k in range(2 * i + 1):
                print("*", end="")
            print()
number=int(input("Please enter the number of rows: "))            
startriangle(number)