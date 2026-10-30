import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

PROCESSED_DIR = Path("Goodreads_rewiew_test_set")
MIN_COMMON_BOOKS = 2
MIN_NEIGHBOR_SIMILARITY = 0.9

VECTOR_FIELD = "emo_lex_emotion_vector"
# Change to "nrc_eil_emotion_vector" to use NRC-EIL.

EMOTION_KEYS = (
    "Anger", "Anticipation", "Disgust", "Fear",
    "Joy", "Sadness", "Surprise", "Trust",
)

def load_user_histories_with_emotion(
    processed_dir=PROCESSED_DIR,
    vector_field=VECTOR_FIELD,
):
    """
    Return:
      user_ratings: {user_id: {book_id: rating}}
      user_emotion_vectors: {user_id: {book_id: {emotion: value}}}

    Both lookups use only each user's history.
    """
    valid_vector_fields = {
        "emo_lex_emotion_vector",
        "nrc_eil_emotion_vector",
    }
    if vector_field not in valid_vector_fields:
        raise ValueError(
            f"vector_field must be one of {sorted(valid_vector_fields)}"
        )

    user_ratings = {}
    user_emotion_vectors = {}

    for file_path in Path(processed_dir).glob("*.json"):
        try:
            with file_path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            user_id = str(data.get("user_id", file_path.stem))
            history = data.get("history", [])

            ratings = {}
            emotion_vectors = {}

            for item in history:
                book_id = str(item["book_id"])
                ratings[book_id] = float(item["rating"])

                vector = item.get(vector_field)
                if vector is not None:
                    emotion_vectors[book_id] = {
                        emotion: float(vector.get(emotion, 0.0))
                        for emotion in EMOTION_KEYS
                    }

            user_ratings[user_id] = ratings
            user_emotion_vectors[user_id] = emotion_vectors

        except Exception as error:
            print(f"Error reading {file_path.name}: {error}")

    return user_ratings, user_emotion_vectors



def load_user_histories(processed_dir=PROCESSED_DIR):
    """
    Return {user_id: {book_id: rating}}, using only each file's history.
    User and book IDs are converted to strings.
    """
    user_ratings = {}

    for file_path in Path(processed_dir).glob("*.json"):
        try:
            with file_path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            user_id = str(data.get("user_id", file_path.stem))
            history = data.get("history", [])

            user_ratings[user_id] = {
                str(item["book_id"]): float(item["rating"])
                for item in history
            }

        except Exception as error:
            print(f"Error reading {file_path.name}: {error}")

    return user_ratings


def calculate_pearson_correlation_with_emotion(
    vectors_u,
    vectors_v,
    min_common_books=MIN_COMMON_BOOKS,
):
    """
    Calculate Pearson correlation over aligned emotion values
    for books both users reviewed.
    """
    shared_books = sorted(set(vectors_u) & set(vectors_v))

    if len(shared_books) < min_common_books:
        return 0.0

    u_vals = []
    v_vals = []

    for book_id in shared_books:
        for emotion in EMOTION_KEYS:
            u_vals.append(vectors_u[book_id].get(emotion, 0.0))
            v_vals.append(vectors_v[book_id].get(emotion, 0.0))

    mean_u = sum(u_vals) / len(u_vals)
    mean_v = sum(v_vals) / len(v_vals)

    numerator = sum(
        (u - mean_u) * (v - mean_v)
        for u, v in zip(u_vals, v_vals)
    )
    denom_u = sum((u - mean_u) ** 2 for u in u_vals)
    denom_v = sum((v - mean_v) ** 2 for v in v_vals)

    if denom_u == 0 or denom_v == 0:
        return 0.0

    return numerator / (math.sqrt(denom_u) * math.sqrt(denom_v))

def calculate_pearson_correlation(ratings_u, ratings_v):
    """Calculate Pearson correlation using books both users rated."""
    shared_books = set(ratings_u) & set(ratings_v)

    if len(shared_books) < 2:
        return 0.0

    u_vals = [ratings_u[book] for book in shared_books]
    v_vals = [ratings_v[book] for book in shared_books]

    mean_u = sum(u_vals) / len(u_vals)
    mean_v = sum(v_vals) / len(v_vals)

    numerator = sum(
        (u - mean_u) * (v - mean_v)
        for u, v in zip(u_vals, v_vals)
    )
    denom_u = sum((u - mean_u) ** 2 for u in u_vals)
    denom_v = sum((v - mean_v) ** 2 for v in v_vals)

    if denom_u == 0 or denom_v == 0:
        return 0.0

    return numerator / (math.sqrt(denom_u) * math.sqrt(denom_v))


def build_similarity_lookup(user_ratings, with_emotion = False):
    """
    Compare every pair of users and return each user's neighbors,
    sorted from highest Pearson correlation to lowest.
    """
    user_ids = list(user_ratings)
    similarity_lookup = {user_id: {} for user_id in user_ids}

    print("Computing user-to-user similarities...")

    for user_a, user_b in combinations(user_ids, 2):
        shared_books = set(user_ratings[user_a]) & set(user_ratings[user_b])

        if len(shared_books) < MIN_COMMON_BOOKS:
            continue

        if with_emotion:
            similarity = calculate_pearson_correlation_with_emotion(
                user_ratings[user_a],
                user_ratings[user_b],
            )
        else:       
            similarity = calculate_pearson_correlation(
                user_ratings[user_a],
                user_ratings[user_b],
            )

        similarity_lookup[user_a][user_b] = similarity
        similarity_lookup[user_b][user_a] = similarity

    return {
        user_id: sorted(
            neighbors.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        for user_id, neighbors in similarity_lookup.items()
    }


def average_recommender(user_ratings, similarity_lookup, results_dir=Path("Goodreads_review_test_results/with_emotion_avg_9")):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for file_path in PROCESSED_DIR.glob("*.json"):
        try:
            with file_path.open("r", encoding="utf-8") as file:
                user_data = json.load(file)

            user_id = str(user_data.get("user_id", file_path.stem))
            neighbors = similarity_lookup.get(user_id, [])

            for recommendation in user_data.get("recommendations", []):
                book_id = str(recommendation["book_id"])
                neighbor_ratings = []

                for neighbor_id, similarity in neighbors:
                    # Neighbors are sorted from highest to lowest similarity.
                    if similarity < MIN_NEIGHBOR_SIMILARITY:
                        break

                    neighbor_history = user_ratings.get(str(neighbor_id), {})
                    if book_id in neighbor_history:
                        neighbor_ratings.append(neighbor_history[book_id])

                if neighbor_ratings:
                    recommendation["predicted_rating"] = (
                        sum(neighbor_ratings) / len(neighbor_ratings)
                    )
                else:
                    recommendation["predicted_rating"] = None

            output_path = results_dir / f"{file_path.stem}_predicted.json"

            with output_path.open("w", encoding="utf-8") as file:
                json.dump(user_data, file, indent=2, ensure_ascii=False)

            print(f"Saved predictions to {output_path}")

        except Exception as error:
            print(f"Error processing {file_path.name}: {error}")


def base_recommender(user_ratings, similarity_lookup, results_dir=Path("Goodreads_review_test_results/with_emotion_base_1")):
    #user_ratings = load_user_histories()
    #similarity_lookup = build_similarity_lookup(user_ratings)

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for file_path in PROCESSED_DIR.glob("*.json"):
        try:
            with file_path.open("r", encoding="utf-8") as file:
                user_data = json.load(file)

            user_id = str(user_data.get("user_id", file_path.stem))
            neighbors = similarity_lookup.get(user_id, [])

            for recommendation in user_data.get("recommendations", []):
                book_id = str(recommendation["book_id"])
                predicted_rating = None

                # Neighbors are already sorted from highest to lowest
                # Pearson correlation.
                for neighbor_id, similarity in neighbors:
                    neighbor_history = user_ratings.get(str(neighbor_id), {})

                    if book_id in neighbor_history:
                        predicted_rating = neighbor_history[book_id]
                        break

                recommendation["predicted_rating"] = predicted_rating

            output_path = results_dir / f"{file_path.stem}_predicted.json"

            with output_path.open("w", encoding="utf-8") as file:
                json.dump(user_data, file, indent=2, ensure_ascii=False)

            print(f"Saved predictions to {output_path}")

        except Exception as error:
            print(f"Error processing {file_path.name}: {error}")


def using_emotion_vectors():
    user_ratings, user_emotion_vectors = load_user_histories_with_emotion()
    similarity_lookup = build_similarity_lookup(user_emotion_vectors, with_emotion=True)
    average_recommender(user_ratings, similarity_lookup)

def using_ratings():
    user_ratings = load_user_histories()
    similarity_lookup = build_similarity_lookup(user_ratings)
    average_recommender(user_ratings, similarity_lookup)