def password(str1):
    countc=0
    countl=0
    flag=True
    if len(str1)>8:
        for x in str1:
            if x.isupper():
                countc+=1
            elif x.islower():
                countl+=1
            elif x==" ":
                flag=False
        if flag and countc>0 and countl>0:
            return "password is correct"
        else:
            return "password is incorrect"

passInput=input("Please enter the password: ")
password(passInput)
