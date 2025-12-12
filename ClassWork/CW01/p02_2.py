
str1=input('Please enter the first String: ')
str2=input('Please enter the second String: ')
str3=str1+str2
len1=len(str1)
len2=len(str2)
len3=len(str3)
print(f"The sum of length of str1 and str2: {len1+len2}")
print(f"The length of str1+str2: {len3}")


"""
first_string=input("please enter your first string: ")
second_string=input ("please enter your second string: ")
print(f"len of first string is : {len(first_string)}")
print(f"len of second string is : {len(second_string)}")
print(f"len of sum string is : {len(second_string+first_string)}")
"""


"""
str1=input('Please enter the first String: ')
str2=input('Please enter the second String: ')
len2str=len(str1+str2)
print(len2str)
"""


"""
first_string=input("please enter your first string: ")
second_string=input ("please enter your second string: ")
print(f"len of first string is : {len(first_string.replace(" " ,""))}")
print(f"len of second string is : {len(second_string.replace(" ",""))}")
print(f"len of sum string is : {len((second_string+first_string).replace(" ",""))}")
"""


"""
a="hello"
b="parsa"
print(f"A Length Is : {len(a)} And B Length Is: {len(b)}")
c=a+" "+b
print(f"C Value Is : {c}")
print(f"C Length Is: {len(c)}")
print("=======================================")
print(f"First Character Is: {c[0]}")
"""


"""
str1=input('Please enter the first String: ')
str2=input('Please enter the second String: ')
strMulti=str1*str2
strSum=str1+str2
strMinus=str1-str2
strDivision=str1/str2
strFloat=float(str1)
strInt=int(str1)
print(strSum)
print(strMinus)
print(strMulti)
print(strDivision)
print(strFloat)
print(strInt)
# TypeError: unsupported operand type(s) for +: 'int' and 'str'
# TypeError: unsupported operand type(s) for /: 'int' and 'str'
# استفاده از اپراتور ها بین دو تا ناهمجنس باعث خطا میشه
"""