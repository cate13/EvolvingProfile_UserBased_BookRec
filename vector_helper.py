from typing import List, Union
import numpy as np
import json

def _load_book_data(file_path = "Goodreads_data/vectors/description_emotion_vectors.jsonl"):
    book_map = {}
    print("loading vec data")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                book_id = record.get("book_id")
                if book_id:
                    # Store the whole record (or just the vectors) keyed by ISBN
                    book_map[book_id] = record
                    #print(isbn)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found.")
    return book_map


BASE_BOOK_DATA_CACHE = _load_book_data()


def bilinear_pool(emotion_dict):
    emotions = list(emotion_dict.keys())

    vector = np.array([
        emotion_dict[emotion]
        for emotion in emotions
    ])

    outer_product = np.outer(vector, vector)

    bilinear = {}

    for i in range(len(emotions)):
        for j in range(i, len(emotions)):
            emotion1 = emotions[i]
            emotion2 = emotions[j]

            if emotion1 == emotion2:
                label = emotion1
            else:
                label = f"{emotion1}-{emotion2}"

            bilinear[label] = outer_product[i, j]

    # Normalize the 36-dimensional vector using L2 normalization
    values = np.array(list(bilinear.values()))

    norm = np.linalg.norm(values)

    if norm != 0:
        values = values / norm

    # Put the normalized values back with their labels
    bilinear = dict(zip(bilinear.keys(), values))

    return bilinear

def average_vectors(vectors: List[Union[List[float], dict]]):

    """
    Averages a list of vectors.
    
    Works for:
        - List[float]  (e.g., tf_idf)
        - Dict[str, float] (e.g., emotion_intensity, emotion, empath)

    Returns:
        Averaged vector (same structure as input)
    """

    if not vectors:
        raise ValueError("Vector list is empty.")
    
    vectors = [v for v in vectors if v is not None]

    first = vectors[0]

    # Case 1: List-based vector (e.g., tf_idf)
    if isinstance(first, list):
        return np.mean(np.array(vectors), axis=0).tolist()

    # Case 2: NumPy-based vector
    elif isinstance(first, np.ndarray):
        # We average along the rows (axis 0)
        return np.mean(np.array(vectors), axis=0)
    
    # Case 3: Dict-based vector (emotion/empath)
    elif isinstance(first, dict):
        keys = first.keys()

        # Safety check
        if not all(v.keys() == keys for v in vectors):
            raise ValueError("All dict vectors must have same keys.")

        return {
            key: sum(v[key] for v in vectors) / len(vectors)
            for key in keys
        }

    else:
        raise TypeError(f"Unsupported vector type. {vectors}")

def cosine_similarity(vec1, vec2):
    if type(vec1) is not type(vec2):
        raise ValueError("Both vectors must be the same type.")

    # Case 1: List vectors (e.g., tf_idf)
    if isinstance(vec1, list):
        if len(vec1) != len(vec2):
            raise ValueError("Both list vectors must have same length.")

        v1 = np.array(vec1, dtype=float)
        v2 = np.array(vec2, dtype=float)

    # Case 2: Dict vectors (e.g., emotion/empath)
    elif isinstance(vec1, dict):
        if set(vec1.keys()) != set(vec2.keys()):
            raise ValueError("Both dict vectors must have same keys.")

        # Ensure consistent ordering
        keys = sorted(vec1.keys())
        v1 = np.array([vec1[k] for k in keys], dtype=float)
        v2 = np.array([vec2[k] for k in keys], dtype=float)
    elif isinstance(vec1, np.ndarray):
        if vec1.shape != vec2.shape:
            raise ValueError("Both numpy vectors must have the same shape.")
        v1 = vec1
        v2 = vec2
    else:
        raise TypeError(f"Unsupported vector type. {vec1} {vec2}")

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(v1, v2) / (norm1 * norm2))

def reduce_emotion_vector(base_emotion_vector):
    combined = {
        "Joy_Sadness": base_emotion_vector.get("Joy", 0) - base_emotion_vector.get("Sadness", 0),
        "Trust_Disgust": base_emotion_vector.get("Trust", 0) - base_emotion_vector.get("Disgust", 0),
        "Fear_Anger": base_emotion_vector.get("Fear", 0) - base_emotion_vector.get("Anger", 0),
        "Surprise_Anticipation": base_emotion_vector.get("Surprise", 0) - base_emotion_vector.get("Anticipation", 0)
    }
    return combined

def get_book_vector_by_id(book_id: str, vector_type: str):
    book_id = int(book_id)
    record = BASE_BOOK_DATA_CACHE.get(book_id)

    if not record:
        # Better to return None or raise a specific KeyError if ISBN isn't found
        raise KeyError(f"{book_id} not found in database.")

    vector = record.get(vector_type)
    
    if vector is None:
        raise ValueError(f"Vector type '{vector_type}' missing for book: {book_id}")

    return vector


# if __name__ == "__main__":
#     isbn_1 = "18392937"
#     isbn_2 = "24834186"
#     isbn_3 = "20419142"
#     isbn_list = [isbn_1, isbn_2, isbn_3]
#     for i in isbn_list:
#         print(get_book_vector_by_id(i, "emo_lex"))
#         print(get_book_vector_by_id(i, "nrc_eil"))