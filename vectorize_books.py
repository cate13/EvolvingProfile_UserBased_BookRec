import json
from pathlib import Path
from tqdm import tqdm

from VectorMakers.EmotionVectorMaker import EmotionVectorMaker


# File paths
BOOK_IDS_FILE = Path("Goodreads_book_vectors/missing_book_ids.txt")
INPUT_FILE = Path("Goodreads_Data/goodreads_books_mystery_thriller_crime.json")
OUTPUT_FILE = Path("Goodreads_book_vectors/mystery_thriller_crime_description_emotion_vectors.jsonl")
NOT_FOUND_FILE = Path("Goodreads_book_vectors/book_ids_not_found.txt")


def load_book_ids():
    """Load the book IDs that need to be vectorized."""
    with open(BOOK_IDS_FILE, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def vectorize_books():
    missing_book_ids = load_book_ids()
    print(f"Loaded {len(missing_book_ids):,} book IDs to process.")

    # Create one instance of each vector maker
    emo_lex_emotion_vector_maker = EmotionVectorMaker(use_intensity=False)
    nrc_eil_emotion_vector_maker = EmotionVectorMaker()

    # Track processing results
    found_book_ids = set()
    vectorized_books = 0
    skipped_no_description = 0
    failed_books = 0

    # Read the source JSONL and save matching vectors
    with open(INPUT_FILE, "r", encoding="utf-8") as input_file, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:

        for line in tqdm(input_file, desc="Vectorizing missing books"):
            if not line.strip():
                continue

            try:
                book = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"\nCould not parse a book record: {e}")
                continue

            book_id = book.get("book_id")

            if book_id is None:
                continue

            book_id = str(book_id)

            # Only process books in the missing-ID list
            if book_id not in missing_book_ids:
                continue

            found_book_ids.add(book_id)

            description = book.get("description", "")

            # Skip books without a usable description
            if not isinstance(description, str) or not description.strip():
                skipped_no_description += 1
                continue

            try:
                emo_lex_vector = (
                    emo_lex_emotion_vector_maker.getEmotionVector(description)
                )
                nrc_eil_vector = (
                    nrc_eil_emotion_vector_maker.getEmotionVector(description)
                )

                # Match the structure of description_emotion_vectors.jsonl
                vectorized_book = {
                    "book_id": int(book_id) if book_id.isdigit() else book_id,
                    "emo_lex": emo_lex_vector,
                    "nrc_eil": nrc_eil_vector,
                }

                output_file.write(
                    json.dumps(vectorized_book, ensure_ascii=False) + "\n"
                )
                vectorized_books += 1

            except Exception as e:
                failed_books += 1
                print(f"\nFailed to vectorize book {book_id} ({book.get('title')}): {e}")

    # Requested IDs not present anywhere in the source file
    not_found_book_ids = sorted(missing_book_ids - found_book_ids)

    with open(NOT_FOUND_FILE, "w", encoding="utf-8") as f:
        for book_id in not_found_book_ids:
            f.write(f"{book_id}\n")

    print("\nFinished!")
    print(f"Book IDs requested: {len(missing_book_ids):,}")
    print(f"Book IDs found in source: {len(found_book_ids):,}")
    print(f"Successfully vectorized: {vectorized_books:,}")
    print(f"Skipped (missing description): {skipped_no_description:,}")
    print(f"Failed to vectorize: {failed_books:,}")
    print(f"Book IDs not found in source: {len(not_found_book_ids):,}")
    print(f"Vectors saved to: {OUTPUT_FILE}")
    print(f"Not-found IDs saved to: {NOT_FOUND_FILE}")


if __name__ == "__main__":
    vectorize_books()