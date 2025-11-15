from FunctionsFolder.FunctionsFile import Fcn_class
name=input("Please enter the name: ")
FlagHealth, FlagFault, FlagNone = Fcn_class(name)
print(f"Flag_Health is {FlagHealth}, Flag_Fault is: {FlagFault}, FlagNone is: {FlagNone}")




"""
flag_heallty=0
flag_fault=0
flag_none=0
name=input("please enter name: ")
def check_name(name):
    if name=="class1":
        print("the data is heallty")
        flag_heallty=1
    elif name=="class2":
        print("the data is fault")
        flag_fault=1
    else:
        print("the data is none")
        flag_none=1
    return flag_heallty, flag_fault, flag_none
result=check_name()
print(f"flag: {result}")
"""


"""
def Func_Health_Check(className):
    flag_health=False
    flag_fault=False
    flag_None=False
    if(className=="class1"):
        flag_health=True
        return "Data Is Correct","FlagHealth",flag_health
    elif(className=="class2"):
        flag_fault=True
        return "Data Is InCorrect","FlagFault",flag_fault
    else:
        flag_None=True
        return "Data Is Not Valid","FlagNone",flag_None

from FunctionFolder.FunctionsFile import Func_Health_Check as fhc
print("Enter Class Name")
className=input()
classHeath,flagName,flagStatus=fhc(className)
print(f"ClassHealth Is : {classHeath}")
print(f"Flag Name Is : {flagName}")
print(f"Flag Status Is : {flagStatus}")
"""
