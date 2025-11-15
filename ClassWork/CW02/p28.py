import json
persons={"name":"ali","age":20,"city":"tehran"}
with open(r"E:\Maktab\Artificial Intelligence\Programming\ClassWork\CW3\user.json",
            "w", encoding="utf-8") as f1:
    json.dump(persons,f1)

with open(r"E:\Maktab\Artificial Intelligence\Programming\ClassWork\CW3\user.json",
            "r", encoding="utf-8") as f2:
    print(json.load(f2))
