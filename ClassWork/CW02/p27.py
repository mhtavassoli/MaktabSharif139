try:
    with open(r"E:\Maktab\Artificial Intelligence\Programming\ClassWork\CW3\notes.txt",
            "r", encoding="utf-8") as f1:
        lines = f1.readlines()

    cleaned_lines = []
    for line in lines:
        # Remove extra spaces from the beginning, end and between words
        cleaned_line = " ".join(line.split())
        cleaned_lines.append(cleaned_line)
        
    with open(r"E:\Maktab\Artificial Intelligence\Programming\ClassWork\CW3\notes_clean.txt",
            "w", encoding="utf-8") as f2:
        for line in cleaned_lines:
            f2.write(line + '\n')

    print("Writing has been done in 'notes_clean.txt'.")

except FileNotFoundError:
    print("Error, file 'notes.txt' did not find!")
except Exception as e:
    print(f"Error, in processing file: {e}")
