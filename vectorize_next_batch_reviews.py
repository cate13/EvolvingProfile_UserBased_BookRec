import json
from pathlib import Path
from tqdm import tqdm

from VectorMakers.EmotionVectorMaker import EmotionVectorMaker


# Directories
BASE_DIR = Path("Goodreads_Data")

INPUT_DIRS = [
    BASE_DIR / "1st4000users",
    BASE_DIR / "2nd4000users",
    BASE_DIR / "3rd4000users",
]

MISSING_USERS_FILE = BASE_DIR / "users_missing_review_vectors_100.txt"
OUTPUT_DIR = Path("Goodreads_review_vectors_larger")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_missing_user_ids():
    """Load user IDs that do not already have review vectors."""
    with open(MISSING_USERS_FILE, "r", encoding="utf-8") as f:
        user_ids = {
            line.strip()
            for line in f
            if line.strip()
        }

    print(f"Loaded {len(user_ids)} user IDs to process.")
    return user_ids


def vectorize_reviews():
    # Load target user IDs
    missing_user_ids = load_missing_user_ids()

    # Create one EmotionVectorMaker instance for all reviews
    emo_lex_emotion_vector_maker = EmotionVectorMaker(use_intensity=False)
    nrc_eil_emotion_vector_maker = EmotionVectorMaker()

    # Find every JSON file across all three input directories
    json_files = []

    for input_dir in INPUT_DIRS:
        if not input_dir.exists():
            print(f"Warning: Input directory does not exist: {input_dir}")
            continue

        json_files.extend(input_dir.glob("*.json"))

    json_files = sorted(json_files)

    print(f"Found {len(json_files)} JSON files across the input directories.")

    total_reviews = 0
    failed_reviews = 0
    skipped_reviews = 0
    matched_files = 0
    processed_users = set()

    for file_index, input_file in enumerate(
        tqdm(json_files, desc="Processing files"), start=1
    ):
        # Load reviews from the current JSON file
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                reviews = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"\nCould not read {input_file}: {e}")
            continue

        # Keep only reviews belonging to users on the missing-ID list
        target_reviews = [
            review
            for review in reviews
            if str(review.get("user_id", "")) in missing_user_ids
        ]

        if not target_reviews:
            continue

        matched_files += 1
        vectorized_reviews = []

        for review in target_reviews:
            user_id = review.get("user_id")
            book_id = review.get("book_id")
            review_text = review.get("review_text", "")

            # Skip reviews without text
            if not isinstance(review_text, str) or not review_text.strip():
                skipped_reviews += 1
                continue

            try:
                emo_lex_emotion_vector = (
                    emo_lex_emotion_vector_maker.getEmotionVector(review_text)
                )
                nrc_eil_emotion_vector = (
                    nrc_eil_emotion_vector_maker.getEmotionVector(review_text)
                )

                vectorized_reviews.append({
                    "user_id": user_id,
                    "book_id": book_id,
                    "emo_lex_emotion_vector": emo_lex_emotion_vector,
                    "nrc_eil_emotion_vector": nrc_eil_emotion_vector,
                })

                total_reviews += 1
                processed_users.add(str(user_id))

            except Exception as e:
                failed_reviews += 1
                print(
                    f"\nFailed to vectorize review "
                    f"(user_id={user_id}, book_id={book_id}): {e}"
                )

        # Save only files containing vectorized target-user reviews.
        # Use the original filename.
        if vectorized_reviews:
            output_file = OUTPUT_DIR / input_file.name

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(vectorized_reviews, f, ensure_ascii=False)

            print(
                f"\nSaved {len(vectorized_reviews)} vectorized reviews "
                f"from {input_file.name}"
            )

    # Report users that were not found in the input data
    users_not_found = missing_user_ids - {
        str(user_id) for user_id in processed_users
    }

    print("\nFinished!")
    print(f"Target users: {len(missing_user_ids)}")
    print(f"Users successfully processed: {len(processed_users)}")
    print(f"Target users not successfully processed: {len(users_not_found)}")
    print(f"Files containing target-user reviews: {matched_files}")
    print(f"Successfully vectorized reviews: {total_reviews}")
    print(f"Failed reviews: {failed_reviews}")
    print(f"Skipped reviews without text: {skipped_reviews}")
    print(f"Output directory: {OUTPUT_DIR}")

    if users_not_found:
        print("\nExample users not successfully processed:")
        for user_id in sorted(users_not_found)[:20]:
            print(f"  {user_id}")


if __name__ == "__main__":
    vectorize_reviews()