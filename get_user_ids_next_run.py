import csv
from pathlib import Path

# Directories and files
CSV_FILE = Path("Goodreads_Data/overlapping_user_pairs_100under600.csv")
VECTORS_DIR = Path("Goodreads_review_vectors")
OUTPUT_FILE = Path("Goodreads_Data/users_missing_review_vectors_100.txt")


def find_missing_user_vectors():
    # Read all user IDs from the CSV
    csv_user_ids = set()

    with open(CSV_FILE, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Adjust these column names if your CSV uses different ones
            for column in ("user_id", "user_id_1", "user_id_2"):
                if column in row and row[column].strip():
                    csv_user_ids.add(row[column].strip())

    # Get user IDs from JSON filenames in the vectors directory
    vector_user_ids = {
        path.stem
        for path in VECTORS_DIR.glob("*.json")
    }

    # Find CSV user IDs that don't have a corresponding vector file
    missing_user_ids = sorted(csv_user_ids - vector_user_ids)

    # Save the missing IDs to a text file
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for user_id in missing_user_ids:
            f.write(user_id + "\n")

    # Print summary
    print(f"Unique user IDs in CSV: {len(csv_user_ids)}")
    print(f"User IDs with vector files: {len(csv_user_ids & vector_user_ids)}")
    print(f"User IDs missing vector files: {len(missing_user_ids)}")
    print(f"Missing IDs saved to: {OUTPUT_FILE}")

    return missing_user_ids


if __name__ == "__main__":
    find_missing_user_vectors()