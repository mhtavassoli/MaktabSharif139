import random
TrueValue=random.randint(1,100)
# TrueValue=int(input('Please enter the random Number between 1 to 100: '))
print(f"The random number is: {TrueValue}")

steps=0
geuss=0
guessValue=int(input('Please enter the random Number between 1 to 100 to guess: '))
while geuss==0:
	if guessValue==TrueValue:
		geuss=1
		steps+=1
		print(f"The steps is {steps}")
	elif guessValue<TrueValue:
		steps+=1
		guessValue=int(input('Please enter the greater than before Number between 1 to 100 to guess: '))
	elif guessValue>TrueValue:
		steps+=1
		guessValue=int(input('Please enter the smaller than before Number between 1 to 100 to guess: '))
		

	

		




