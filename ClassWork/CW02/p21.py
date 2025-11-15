def CounterEspecialCharacter(char):
    while True:
        str=input("Please enter the new String or 'exit' to end: ")
        if str=="exit":
            print("End!")
            break
        else:
            counter=0
            for i in str:
                if i==char:
                    counter+=1
            print(f"The numbers of character '{char}' in string '{str}' is equal to {counter}.")
            
CounterEspecialCharacter("e")