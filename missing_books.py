import json
from pathlib import Path
from tqdm import tqdm


# Directories
REVIEW_VECTORS_DIR = Path("Goodreads_review_vectors")
BOOK_VECTORS_DIR = Path("Goodreads_book_vectors")

# Output file
OUTPUT_FILE = Path("missing_book_ids.txt")


def find_missing_books():
    # Collect all unique book IDs from the review vector JSON files
    review_book_ids = set()

    review_files = list(REVIEW_VECTORS_DIR.glob("*.json"))

    print(f"Reading {len(review_files)} review vector files...")

    for filepath in tqdm(review_files, desc="Review vector files"):
        with open(filepath, "r", encoding="utf-8") as f:
            reviews = json.load(f)

        for review in reviews:
            book_id = str(review["book_id"])
            review_book_ids.add(book_id)

    print(f"Unique book IDs in review vectors: {len(review_book_ids)}")

    # Collect all unique book IDs from the book vector JSONL files
    book_vector_ids = set()

    book_files = list(BOOK_VECTORS_DIR.glob("*.jsonl"))

    print(f"\nReading {len(book_files)} book vector files...")

    for filepath in tqdm(book_files, desc="Book vector files"):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                book = json.loads(line)
                book_id = str(book["book_id"])
                book_vector_ids.add(book_id)

    print(f"Unique book IDs in book vectors: {len(book_vector_ids)}")

    # Find books present in review vectors but missing from book vectors
    missing_book_ids = review_book_ids - book_vector_ids

    print(f"\nMissing book IDs: {len(missing_book_ids)}")

    # Save the missing IDs to a text file, one ID per line
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for book_id in sorted(missing_book_ids):
            f.write(f"{book_id}\n")

    print(f"Missing book IDs saved to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    find_missing_books()