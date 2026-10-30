import json
import random
from pathlib import Path

# Set your input and output directory paths
INPUT_DIR = Path("Goodreads_review_test_users")       # Folder containing your original user JSON files
OUTPUT_DIR = Path("Goodreads_rewiew_test_set")      # Folder to save the updated JSON files

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def process_user_files():
    processed_count = 0
    dropped_count = 0

    for file_path in INPUT_DIR.glob("*.json"):
        # Load user interaction data (expected to be a list of book dictionaries)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                user_books = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            continue

        if not isinstance(user_books, list) or len(user_books) == 0:
            continue

        # Extract user_id from the first record (assuming all entries in the file share the same user_id)
        user_id = user_books[0].get("user_id", file_path.stem)

        # Randomly shuffle the user's books to ensure a random split
        random.shuffle(user_books)

        # Calculate split index for 40% history / 60% recommendations
        split_index = int(len(user_books) * 0.4)

        history = user_books[:split_index]
        recommendations = user_books[split_index:]

        # Drop users with less than 4 books in history
        if len(history) < 4:
            dropped_count += 1
            continue

        # Structure the new JSON format
        updated_user_data = {
            "user_id": user_id,
            "history": history,             # 40% of their books (used to build profile)
            "recommendations": recommendations # 60% of their books (used as ground truth/targets)
        }

        # Save to the output directory
        output_file_path = OUTPUT_DIR / file_path.name
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(updated_user_data, f, indent=2)
        
        processed_count += 1

    print(f"Processing complete!")
    print(f"Successfully processed and saved: {processed_count} users")
    print(f"Dropped due to < 4 history books: {dropped_count} users")

if __name__ == "__main__":
    process_user_files()