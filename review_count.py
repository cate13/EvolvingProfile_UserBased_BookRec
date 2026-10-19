import json
from pathlib import Path

INPUT_DIR = Path("Goodreads_data/1st10000users")


def count_reviews():
    json_files = sorted(INPUT_DIR.glob("*.json"))

    total_reviews = 0

    print(f"Found {len(json_files)} JSON files.\n")

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            reviews = json.load(f)

        review_count = len(reviews)
        total_reviews += review_count

        print(f"{file_path.name}: {review_count} reviews")

    print("\n------------------------------")
    print(f"Total files: {len(json_files)}")
    print(f"Total reviews: {total_reviews}")


if __name__ == "__main__":
    count_reviews()