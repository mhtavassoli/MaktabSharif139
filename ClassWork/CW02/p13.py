def findUserFunction(users, name):
    if name in users:
        return users[name]
    else:
        print(f"There is not person with name of '{name}'.")
        return None
############################################################
users = {
    "Ali": {
        "Name": "Ali",
        "Age": 25,
        "City": "Tehran"
			},
    "Maryam": {
        "Name": "Maryam", 
        "Age": 30,
        "City": "Mashhad"
			},
    "Reza": {
        "Name": "Reza",
        "Age": 22,
        "City": "Esfihan"
    }
	}

print(f"{users["Ali"]}" )
user_info = findUserFunction(users, "Ali")
if user_info:
    print(f"User information: {user_info}")

"""
dict={'name':['ali','hassan','hosein','amir'],
	  'age':['21','23','45','2'],
	  'city':['tehran','isfahan','kerman','shiraz']}
def search(name1):
	if name1 in dict['name']:
		index=dict['name'].index(name1)
		return { dict['name'][index], dict['age'][index], dict['city'][index] }
	else:
		return 'not found'
"""

"""
def detect_information(username):	
	if(username in information):
		print(information[username])
information = {
	"ali89":{"name":"ali","age":89,"city":"Tehran"},
	"hasan21":{"name":"hasan","age":21,"city":"Tehran"},
	"parsa35":{"name":"mahdi","age":35,"city":"Tehran"}
	}
detect_information("hasan21")
"""