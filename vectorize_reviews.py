import json
from pathlib import Path
from tqdm import tqdm

from VectorMakers.EmotionVectorMaker import EmotionVectorMaker


# Directories
INPUT_DIR = Path("Goodreads_data/1st10000users")
OUTPUT_DIR = Path("Goodreads_review_vectors")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def vectorize_reviews():
    # Create one EmotionVectorMaker instance for all reviews
    emo_lex_emotion_vector_maker = EmotionVectorMaker(use_intensity=False)
    nrc_eil_emotion_vector_maker = EmotionVectorMaker()

    # Find every JSON file in the input directory
    json_files = sorted(INPUT_DIR.glob("*.json"))

    print(f"Found {len(json_files)} JSON files.")

    total_reviews = 0
    failed_reviews = 0

    for file_index, input_file in tqdm(enumerate(json_files, start=1), total=len(json_files)):
        print(f"[{file_index}/{len(json_files)}] Processing {input_file.name}")

        # Load reviews from the current JSON file
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                reviews = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Could not read {input_file.name}: {e}")
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
                # TODO: Replace this with the actual method from
                # EmotionVectorMaker.py that vectorizes a string.
                emo_lex_emotion_vector = emo_lex_emotion_vector_maker.getEmotionVector(review_text)
                nrc_eil_emotion_vector = nrc_eil_emotion_vector_maker.getEmotionVector(review_text)

                # Preserve the identifiers alongside the vector
                vectorized_reviews.append({
                    "user_id": user_id,
                    "book_id": book_id,
                    "emo_lex_emotion_vector": emo_lex_emotion_vector,
                    "nrc_eil_emotion_vector": nrc_eil_emotion_vector
                })

                total_reviews += 1

            except Exception as e:
                failed_reviews += 1
                print(
                    f"  Failed to vectorize review "
                    f"(user_id={user_id}, book_id={book_id}): {e}"
                )

        # Save the vectorized reviews using the original filename
        output_file = OUTPUT_DIR / input_file.name

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(vectorized_reviews, f, ensure_ascii=False)

        print(f"  Saved {len(vectorized_reviews)} vectorized reviews.")

    print("\nFinished!")
    print(f"Successfully vectorized reviews: {total_reviews}")
    print(f"Failed reviews: {failed_reviews}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    vectorize_reviews()