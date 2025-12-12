import json

def usersOver20Function():    
    try:
        with open(r"E:\Maktab\Artificial Intelligence\Programming\ClassWork\CW3\users.json",
                    "r", encoding="utf-8") as file:
                    users = json.load(file)
        
        for user in users:
            if user.get("age", 0) > 20:   
                print(user["name"])
                
# اگر کلید "age" وجود داشته باشد، مقدارش برمی‌گردد
# اگر وجود نداشته باشد، مقدار پیش‌فرض (دومین پارامتر) برمی‌گردد

    
    except FileNotFoundError:
        print("Error: file users.json did not find.")
    except json.JSONDecodeError:
        print("Error: file JSON invalid.")
    except Exception as e:
        print(f"Error unknown: {e}")

usersOver20Function()