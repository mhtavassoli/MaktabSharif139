# with open("notes.txt","w") as f:        # E:\Maktab\Artificial Intelligence\Programming
# with open("E:\\Maktab\\Artificial Intelligence\\Programming\\ClassWork\\CW3\\notes.txt","w") as f: 
with open(r"E:\Maktab\Artificial Intelligence\Programming\ClassWork\CW3\notes.txt","w") as f:  # r: raw string
    while True:
        Line=input("Enter a string: ")
        if Line.lower()=="exit":
            break
        else:
            f.write(Line + "\n")
print("Finish!")


"""
flag=True
mymessage=""
while flag:
    text=input("Enter Text:")
    if str.lower(text)=="exit":
        flag=False
    else:
        mymessage+=text+"\n"
with open("notes.txt", "w") as f:
    f.write(mymessage)
"""

"""
mymessage=""
while True:
    Line=input("Enter Text:")
    if str.lower(Line)=="exit":
        break
    else:
        mymessage+=Line+"\n"
with open("notes.txt", "w") as f:
    f.write(mymessage)
"""