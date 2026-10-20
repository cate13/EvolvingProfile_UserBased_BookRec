import json
from pathlib import Path
from tqdm import tqdm

BOOK_IDS_FILE = Path("Goodreads_review_vectors/unique_book_ids.txt")
VECTORS_FILE = Path("Goodreads_book_vectors/description_emotion_vectors.jsonl")
OUTPUT_FILE = Path("Goodreads_book_vectors/missing_book_ids.txt")


def find_missing_book_ids():
    # Load the book IDs that should be present
    with open(BOOK_IDS_FILE, "r", encoding="utf-8") as f:
        expected_book_ids = {
            line.strip() for line in f if line.strip()
        }

    print(f"Book IDs in TXT file: {len(expected_book_ids):,}")

    # Collect book IDs already present in the vector file
    existing_book_ids = set()

    with open(VECTORS_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading vectorized books"):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                book_id = record.get("book_id")

                if book_id is not None:
                    existing_book_ids.add(str(book_id))

            except json.JSONDecodeError as e:
                print(f"\nSkipping invalid JSON line: {e}")

    print(f"Book IDs in vector file: {len(existing_book_ids):,}")

    # Compare IDs as strings to handle numeric/string type differences
    missing_book_ids = sorted(expected_book_ids - existing_book_ids)

    # Save missing IDs, one per line
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for book_id in missing_book_ids:
            f.write(f"{book_id}\n")

    print(f"\nMissing book IDs: {len(missing_book_ids):,}")
    print(f"Saved missing IDs to: {OUTPUT_FILE}")


if __name__ == "__main__":
    find_missing_book_ids()