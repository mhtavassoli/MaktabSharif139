def safe_convert(value,type):
    try:
        if type=="int":
            return int(value)
        elif type=="float":
            return float(value)
        elif type=="str":
            return str(value)
    except:
        return None
    
############################################################################## input way 1             
inputValue=12
inputDataType="int"
output=safe_convert(inputValue,inputDataType)
print(f"outputValue: {output} , dataType: {type(output)}")
############################################################################## input way 2
inputDictionary = {
    "case1": {
        "Value": 12,
        "DataType": "int",
			},
    "case2": {
        "Value": 12, 
        "DataType": "float",
			},
    "case3": {
        "Value": 12,
		"DataType": "str"},
	"case4": {
		"Value": 12,
        "DataType": "bool"}
        }

for i in inputDictionary:
	output=safe_convert(inputDictionary[i]["Value"],inputDictionary[i]["DataType"])
	print(f"outputValue: {output} \t ,dataType: {type(output)}")
