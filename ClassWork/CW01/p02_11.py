from FunctionsFolder.FunctionsFile import Fcn_Days
input=int(input("Please enter the Days: "))
years, monthes, days= Fcn_Days(input)
print(Fcn_Days(input))                                              # (1, 1, 10)
print(f"years is: {years}, monthes is {monthes}, days is {days}")   # years is: 1, monthes is 1, days is 10
