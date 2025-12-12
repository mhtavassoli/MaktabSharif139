import json
with open(r"E:\Maktab\Artificial Intelligence\Programming\ClassWork\CW3\user.json",
            "r", encoding="utf-8") as f1:
    persons=json.load(f1)
print(persons)
persons["city"]=input("Please enter the new city: ")
with open(r"E:\Maktab\Artificial Intelligence\Programming\ClassWork\CW3\user.json",
        "w", encoding="utf-8") as f2:
    json.dump(persons,f2)
print(persons)

    
