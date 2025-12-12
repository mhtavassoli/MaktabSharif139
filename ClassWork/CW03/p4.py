import json
import random
# way 1---------------------------------------------------------------------------------------------
def fuctionPersonsWay1():
    persons = [
        {"Name": "ali", "Score": 20},
        {"Name": "sara", "Score": 18},
        {"Name": "reza", "Score": 16},
        {"Name": "maryam", "Score": 15},
        {"Name": "akbar", "Score": 17},
        {"Name": "kambiz", "Score": 10},
        {"Name": "mitra", "Score": 8},
        {"Name": "nazanin", "Score": 12},
        {"Name": "houshang", "Score": 11},
        {"Name": "hosein", "Score": 17}
    ]
    print(persons)
    return persons

# way 2---------------------------------------------------------------------------------------------
def fuctionPersonsWay2():
    persons = []
    N=int(input("Please enter the number of persons: "))
    for i in range(N):
        NameInput=input("Please enter the name of student: ")
        ScoreInput=int(input("Please enter the score of student: "))
        # persons.append((NameInput, ScoreInput))   # append without label like: [('a', 1), ('b', 2)]
        persons.append({"Name": NameInput, "Score": ScoreInput})
    print(persons)
    return persons

# way 3---------------------------------------------------------------------------------------------
def functionPersonsWay3():
    persons = {"Name": [], "Score": []}
    N=int(input("Please enter the number of persons: "))
    for i in range(N):
        NameInput=input("Please enter the name of student: ")
        ScoreInput=int(input("Please enter the score of student: "))
        persons["Name"].append(NameInput)
        persons["Score"].append(ScoreInput)
    print(persons)
    return persons

# way 4---------------------------------------------------------------------------------------------
def functionPersonsWay4():
    Names = ["ali", "sara", "reza", "maryam", "ahmad", "fatima",
            "hossein", "zahra", "mohammad", "narges"]
    persons = []
    for i in range(10):        
        persons.append({
            "Name": random.choice(Names),
            "Score": random.randint(0, 20)
        })
    print(persons)
    return persons

# way 5---------------------------------------------------------------------------------------------
def functionPersonsWay5():
    Names = ["ali", "sara", "reza", "maryam", "ahmad", "fatima",
            "hossein", "zahra", "mohammad", "narges"]
    persons = {}
    for i in range(1, 11):
        persons[f"person{i}"] = {
            "Name": random.choice(Names),
            "Score": random.randint(0, 20)
        }
    print(persons)
    return persons

# 1-------------------------------------------------------------------------------------------------
def save_scores():
    persons=fuctionPersonsWay1()
    with open(r"E:\Maktab\Artificial Intelligence\VsCodeExplorer\ClassWork\CW4\scores.json",
        "w", encoding="utf-8") as f1:
        json.dump(persons,f1)
    print("Saving done!")
        
# 2-------------------------------------------------------------------------------------------------
def check_scores():
    with open(r"E:\Maktab\Artificial Intelligence\VsCodeExplorer\ClassWork\CW4\scores.json",
            "r", encoding="utf-8") as f1:
        persons=json.load(f1)
    for i in range(len(persons)):
        if persons[i]["Score"]>10:
            print(f"{i}- The person with name '{persons[i]["Name"]}' and "
                  f"score '{persons[i]["Score"]}' : pass")
        else:
            print(f"{i}- The person with name '{persons[i]["Name"]}' and "
                  f"score '{persons[i]["Score"]}': fail")    

# 3-------------------------------------------------------------------------------------------------
save_scores()
check_scores()
