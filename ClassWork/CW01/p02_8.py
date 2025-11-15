def longestname(name1,name2):
    if len(name1)>len(name2):
        return name1
    else:
        return name2
print(longestname(input("please enter your name: "),input("please enter your name :")))


"""
name1=input("enter your name: ")
name2=input("enter your name: ")
def longest_name(name1,name2):
    if len(name1) > len(name2):
        return name1
    elif len(name2) > len(name1):
        return name2
    else:
        return "same"
print(longest_name(name1,name2))
"""


"""
def longestname(name1,name2):
    if len(name1)>len(name2):
        return name1,len(name1)
    elif len(name2)>len(name1):
        return name2,len(name2)
    else:
        return f"{name1} and {name2} has same lens and len is {len(name1)}"
print(longestname(input("please enter your name: "),input("please enter your name :")))
"""
