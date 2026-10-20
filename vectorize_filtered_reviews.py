import csv
import json
from pathlib import Path
from tqdm import tqdm

from VectorMakers.EmotionVectorMaker import EmotionVectorMaker


# Directories
INPUT_DIR = Path("Goodreads_data/1st10000users")
PAIRS_FILE = Path("Goodreads_data/overlapping_user_pairs_100.csv")
OUTPUT_DIR = Path("Goodreads_review_vectors")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_overlapping_users():
    """Load unique user IDs appearing in the overlapping pairs CSV."""
    overlapping_users = set()

    with open(PAIRS_FILE, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            user_1 = row.get("user_id_1")
            user_2 = row.get("user_id_2")

            if user_1:
                overlapping_users.add(user_1.strip())

            if user_2:
                overlapping_users.add(user_2.strip())

    return overlapping_users


def vectorize_reviews():
    # Load users that appear in at least one qualifying pair
    overlapping_users = load_overlapping_users()

    print(f"Loaded {len(overlapping_users):,} unique overlapping users.")

    # Create one EmotionVectorMaker instance for all reviews
    emo_lex_emotion_vector_maker = EmotionVectorMaker(use_intensity=False)
    nrc_eil_emotion_vector_maker = EmotionVectorMaker()

        # Find JSON files belonging to overlapping users only
    # Skip users whose output JSON file already exists
    json_files = sorted(
        file_path
        for file_path in INPUT_DIR.glob("*.json")
        if (
            file_path.stem in overlapping_users
            and not (OUTPUT_DIR / file_path.name).exists()
        )
    )

    # Count users already processed
    already_processed = sum(
        1
        for file_path in INPUT_DIR.glob("*.json")
        if (
            file_path.stem in overlapping_users
            and (OUTPUT_DIR / file_path.name).exists()
        )
    )

    print(f"Already processed: {already_processed:,} users")
    print(f"Remaining to process: {len(json_files):,} users")

    total_reviews = 0
    failed_reviews = 0

    for file_index, input_file in enumerate(
        tqdm(json_files, desc="Vectorizing users"),
        start=1,
    ):
        print(f"[{file_index}/{len(json_files)}] Processing {input_file.name}")
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                reviews = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"\nCould not read {input_file.name}: {e}")
            continue

        vectorized_reviews = []

        for review in reviews:
            user_id = review.get("user_id")
            book_id = review.get("book_id")
            review_text = review.get("review_text", "")

            # Skip reviews without text
            if not isinstance(review_text, str) or not review_text.strip():
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

            except Exception as e:
                failed_reviews += 1
                print(
                    f"\nFailed to vectorize review "
                    f"(user_id={user_id}, book_id={book_id}): {e}"
                )

        # Save using the original filename
        output_file = OUTPUT_DIR / input_file.name

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(vectorized_reviews, f, ensure_ascii=False)

        print(f"  Saved {len(vectorized_reviews)} vectorized reviews.")

    print("\nFinished!")
    print(f"Unique overlapping users in CSV: {len(overlapping_users):,}")
    print(f"User files processed: {len(json_files):,}")
    print(f"Successfully vectorized reviews: {total_reviews:,}")
    print(f"Failed reviews: {failed_reviews:,}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    vectorize_reviews()