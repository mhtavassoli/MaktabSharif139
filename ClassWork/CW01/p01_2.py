name="mhosein"
age=39
color="blue"

# way 1, converting
print("My name is: " + name + ", My age is: " + str(age) + ", My color is: " + color)

# way 2, Using f-string (recommended)
print(f"My name is: {name}, My age is: {age}, My color is: {color}")
# or
message=f"My name is: {name}, My age is: {age}, My color is: {color}"
print(message)

#way 3, Using .format() method
print("My name is: {}, My age is: {}, My color is: {}".format(name, age, color))

    