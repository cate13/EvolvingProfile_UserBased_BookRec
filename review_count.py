import csv
import json
from pathlib import Path

INPUT_DIR = Path("Goodreads_data/1st10000users")
PAIRS_FILE = Path("Goodreads_data/overlapping_user_pairs_100.csv")


def load_overlapping_users():
    """Get unique user IDs appearing in either column of the CSV."""
    users = set()

    with open(PAIRS_FILE, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            user_1 = row.get("user_id_1")
            user_2 = row.get("user_id_2")

            if user_1:
                users.add(user_1.strip())
            if user_2:
                users.add(user_2.strip())

    return users


def count_reviews():
    overlapping_users = load_overlapping_users()

    # Only include files belonging to users in the CSV
    json_files = sorted(
        file_path
        for file_path in INPUT_DIR.glob("*.json")
        if file_path.stem in overlapping_users
    )

    total_reviews = 0

    print(f"Unique users in CSV: {len(overlapping_users):,}")
    print(f"Matching JSON files: {len(json_files):,}\n")

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            reviews = json.load(f)

        review_count = len(reviews)
        total_reviews += review_count

        print(f"{file_path.name}: {review_count} reviews")

    print("\n------------------------------")
    print(f"Total users counted: {len(json_files):,}")
    print(f"Total reviews: {total_reviews:,}")


if __name__ == "__main__":
    count_reviews()