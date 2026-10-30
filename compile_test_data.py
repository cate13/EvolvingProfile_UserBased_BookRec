import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


# Directories
RAW_USER_DIRS = [
    Path("Goodreads_Data/1st4000users"),
    Path("Goodreads_Data/1st10000users"),
    Path("Goodreads_Data/2nd4000users"),
    Path("Goodreads_Data/3rd4000users"),
]

VECTOR_DIRS = [
    Path("Goodreads_review_vectors_larger"),
    Path("Goodreads_review_vectors"),
]

BOOK_VECTORS_FILE = Path(
    "Goodreads_book_vectors/all_description_emotion_vectors.jsonl"
)

OUTPUT_DIR = Path("Goodreads_review_test_set")


def normalize_book_id(book_id):
    """Normalize book IDs so numeric and string IDs match."""
    return str(book_id).strip()


def load_valid_book_ids():
    """Load all book IDs available in the comprehensive book vector file."""
    book_ids = set()

    with BOOK_VECTORS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            book_ids.add(normalize_book_id(record["book_id"]))

    print(f"Loaded {len(book_ids):,} valid book IDs.")
    return book_ids


def load_user_ratings():
    """
    Build a lookup from (user_id, book_id) to rating using the raw
    Goodreads user review files.

    If a user/book pair occurs multiple times, the first rating is kept.
    Conflicting duplicate ratings are counted and reported.
    """
    ratings = {}
    conflicting_ratings = 0
    total_reviews = 0

    for directory in RAW_USER_DIRS:
        if not directory.exists():
            print(f"Warning: Raw user directory not found: {directory}")
            continue

        files = sorted(directory.glob("*.json"))

        for file_path in tqdm(files, desc=f"Loading ratings: {directory.name}"):
            with file_path.open("r", encoding="utf-8") as f:
                reviews = json.load(f)

            for review in reviews:
                total_reviews += 1

                user_id = str(review["user_id"])
                book_id = normalize_book_id(review["book_id"])
                rating = review["rating"]

                key = (user_id, book_id)

                if key not in ratings:
                    ratings[key] = rating
                elif ratings[key] != rating:
                    conflicting_ratings += 1

    print(f"\nRaw reviews processed: {total_reviews:,}")
    print(f"Unique user/book pairs: {len(ratings):,}")
    print(f"Conflicting duplicate ratings: {conflicting_ratings:,}")

    return ratings


def load_and_merge_vectors(valid_book_ids, ratings):
    """
    Load vectorized reviews from both vector directories.

    - Exclude books without a comprehensive book vector.
    - Add the matching raw rating.
    - Merge duplicate user/book pairs, keeping the first vector encountered.
    - Goodreads_review_vectors_larger has priority over Goodreads_review_vectors.
    """
    user_reviews = defaultdict(dict)

    total_vector_reviews = 0
    invalid_book_reviews = 0
    missing_ratings = 0
    duplicate_reviews = 0

    for directory in VECTOR_DIRS:
        if not directory.exists():
            print(f"Warning: Vector directory not found: {directory}")
            continue

        files = sorted(directory.glob("*.json"))

        for file_path in tqdm(files, desc=f"Processing vectors: {directory.name}"):
            with file_path.open("r", encoding="utf-8") as f:
                reviews = json.load(f)

            for review in reviews:
                total_vector_reviews += 1

                user_id = str(review["user_id"])
                book_id = normalize_book_id(review["book_id"])
                key = (user_id, book_id)

                # Only retain books with a comprehensive book vector.
                if book_id not in valid_book_ids:
                    invalid_book_reviews += 1
                    continue

                # Keep only reviews that have a matching raw rating.
                if key not in ratings:
                    missing_ratings += 1
                    continue

                # Deduplicate by user and book.
                # The larger directory is processed first, so its vector
                # is retained if the same pair appears in both directories.
                if book_id in user_reviews[user_id]:
                    duplicate_reviews += 1
                    continue

                # Preserve the original vector fields and add the rating.
                enriched_review = dict(review)
                enriched_review["rating"] = ratings[key]

                user_reviews[user_id][book_id] = enriched_review

    print(f"\nVector reviews processed: {total_vector_reviews:,}")
    print(f"Excluded books without book vectors: {invalid_book_reviews:,}")
    print(f"Excluded reviews without matching ratings: {missing_ratings:,}")
    print(f"Duplicate user/book reviews skipped: {duplicate_reviews:,}")

    return user_reviews


def save_user_reviews(user_reviews):
    """Save one JSON file per user in Goodreads_review_test_set."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_saved_reviews = 0

    for user_id, reviews_by_book in tqdm(
        user_reviews.items(), desc="Saving user reviews"
    ):
        output_file = OUTPUT_DIR / f"{user_id}.json"

        reviews = list(reviews_by_book.values())

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(reviews, f, ensure_ascii=False)

        total_saved_reviews += len(reviews)

    print("\nFinished!")
    print(f"Users saved: {len(user_reviews):,}")
    print(f"Reviews saved: {total_saved_reviews:,}")
    print(f"Output directory: {OUTPUT_DIR}")


def main():
    valid_book_ids = load_valid_book_ids()
    ratings = load_user_ratings()

    user_reviews = load_and_merge_vectors(
        valid_book_ids=valid_book_ids,
        ratings=ratings,
    )

    save_user_reviews(user_reviews)


if __name__ == "__main__":
    main()