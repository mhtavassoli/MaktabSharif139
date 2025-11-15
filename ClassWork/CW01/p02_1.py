total_seconds=int(input("please enter total seconds: "))
hours=total_seconds//3600
remaining_seconds=total_seconds%3600
minutes=remaining_seconds//60
seconds=remaining_seconds%60
print(f"total seconds is : {total_seconds}")
print(f"hours: {hours},minutes: {minutes},seconds: {seconds}")


"""
seconds = int(input(" please input total seconds: "))
print(f"hours is {round(seconds/3600)} and minutes is {round(seconds/60)} and seconds is {seconds%60} ")
"""