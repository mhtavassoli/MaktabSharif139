def DictionaryAnalysis(dictionaryInput):
    if dictionaryInput["Average"]<12:
        print(f"Student {dictionaryInput["Name"]} with {dictionaryInput["Age"]} years"
              f"with Averge={dictionaryInput["Average"]} has not been accepted.")
    else:
        print(f"Student {dictionaryInput["Name"]} with {dictionaryInput["Age"]} years"
              f"with Averge={dictionaryInput["Average"]} has been accepted.")
    return
    
student1 = {
  "Name": "Ali",
  "Age": "16",
  "Average": 12
}
DictionaryAnalysis(student1)
