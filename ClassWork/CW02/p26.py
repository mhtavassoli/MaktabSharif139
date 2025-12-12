try:  
    with open(r"E:\Maktab\Artificial Intelligence\Programming\ClassWork\CW3\notes.txt",
              "r", encoding="utf-8") as f:
        # Way 1
        lines = f.readlines()
        line_count = len(lines)
        char_count = 0
        word_count = 0
        for line in lines:
            char_count += len(line)
            word_count += len(line.split())
        
        """ 
        # way 2
        content = f.read()
        line_count = content.count('\n') + (1 if content else 0)
        char_count = len(content)
        word_count = len(content.split())
        """
        print(f"The number of lines is: {line_count}")
        print(f"The number of Characters is: {char_count}")
        print(f"The number of words is: {word_count}")
        
except FileNotFoundError:
    print("error: file 'notes.txt' didn't find!")
except Exception as e:
    print(f"error in processing file: {e}")