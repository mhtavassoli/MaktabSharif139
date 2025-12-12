import json
import os
from datetime import datetime

def load_notes(filename=r"E:\Maktab\Artificial Intelligence\VsCodeExplorer\HomeWork\HW02\notes.json"):
    """Loading notes from file"""
    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error in loading notes: {e}")
        return []

def save_notes(notes, filename=r"E:\Maktab\Artificial Intelligence\VsCodeExplorer\HomeWork\HW02\notes.json"):
    """save notes in file"""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(notes, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        print(f"Error in save notes: {e}")
        return False

def add_note(notes):  
    """Adding new note"""
    print("\n📝 Adding new note")
    print("=" * 30)
  
    title = input("Title of note: ").strip()
    content = input("Content of note: ").strip()
    
    if not title or not content:
        print("❌ Title and content could not be empty!")
        return notes
    
    note = {
        "title": title,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "content": content
    }
    
    notes.append(note)
    if save_notes(notes):
        print("✅ Note added successfully.!")
    else:
        print("❌ Error savig note!")
    
    return notes

def view_all_notes(notes):
    print("\n📖 All notes")
    print("=" * 50)
    
    if not notes:
        print("📭 There are no notes!")
        return
    
    for i, note in enumerate(notes, 1):
        print(f"{i}. Title: {note['title']}")
        print(f"   Date: {note['date']}")
        print(f"   Content: {note['content']}")
        print("-" * 50)

def search_notes(notes):
    """Search in notes"""
    print("\n🔍 Search in notes")
    print("=" * 40)
    
    if not notes:
        print("📭 There are no notes to search for!")
        return
    
    search_term = input("Search term (part of the title): ").strip().lower()
    
    if not search_term:
        print("❌ Search term cannot be empty.!")
        return
    
    found_notes = [note for note in notes if search_term in note['title'].lower()]
    
    if not found_notes:
        print("❌ No notes found.!")
        return
    
    print(f"\n🔎 Search results for '{search_term}':")
    print("=" * 50)
    
    for i, note in enumerate(found_notes, 1):
        print(f"{i}. Title: {note['title']}")
        print(f"   Date: {note['date']}")
        print(f"   Content: {note['content']}")
        print("-" * 50)

def delete_note(notes):
    """Delete a specific note"""
    print("\n🗑️ Delete note")
    print("=" * 30)
    
    if not notes:
        print("📭 There are no notes to delete!")
        return notes
    
# Show list of notes to select
    print("List of notes :")
    for i, note in enumerate(notes, 1):
        print(f"{i}. {note['title']}")
    
    try:
        choice = int(input("\n Note number to delete: "))
        if 1 <= choice <= len(notes):
            deleted_note = notes.pop(choice - 1)
            if save_notes(notes):
                print(f"✅ Note '{deleted_note['title']}' successfully deleted.!")
            else:
                print("❌ Error in saving changes!")
        else:
            print("❌ Invalid number!")
    except ValueError:
        print("❌ Please enter a number!")
    
    return notes

def show_menu():
    """Show main menu"""
    print("\n" + "=" * 40)
    print("📓 Digital notebook")
    print("=" * 40)
    print("1. Add new note")
    print("2. View all notes")
    print("3. Search in notes")
    print("4. Delete a specific note")
    print("5. exit")
    print("=" * 40)

def main():
    """Main function of the program"""
    notes = load_notes()
    
    while True:
        show_menu()
        
        try:
            choice = input("Please select the desired option (1-5): ").strip()
            
            if choice == "1":
                notes = add_note(notes)
            elif choice == "2":
                view_all_notes(notes)
            elif choice == "3":
                search_notes(notes)
            elif choice == "4":
                notes = delete_note(notes)
            elif choice == "5":
                print("\n👋 Thank you for your use! Goodbye!")
                break
            else:
                print("❌ Invalid option! Please enter a number between 1 and 5.")
        
        except KeyboardInterrupt:
            print("\n\n👋 The application was stopped by the user!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()