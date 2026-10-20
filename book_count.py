import json
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = Path("Goodreads_review_vectors")
OUTPUT_FILE = Path("Goodreads_review_vectors/unique_book_ids.txt")


def collect_unique_book_ids():
    json_files = sorted(INPUT_DIR.glob("*.json"))
    unique_book_ids = set()

    print(f"Found {len(json_files):,} JSON files.")

    for file_path in tqdm(json_files, desc="Collecting book IDs"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reviews = json.load(f)

            for review in reviews:
                book_id = review.get("book_id")

                if book_id is not None:
                    unique_book_ids.add(str(book_id))

        except (json.JSONDecodeError, OSError) as e:
            print(f"\nCould not process {file_path.name}: {e}")

    # Sort IDs for consistent output
    sorted_book_ids = sorted(unique_book_ids)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for book_id in sorted_book_ids:
            f.write(f"{book_id}\n")

    print("\nFinished!")
    print(f"Unique book IDs: {len(unique_book_ids):,}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    collect_unique_book_ids()