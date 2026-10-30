import json
from pathlib import Path


def load_book_vector_lookup(jsonl_path):
    """Return {str(book_id): {"emo_lex": ..., "nrc_eil": ...}}."""
    book_vectors = {}

    with Path(jsonl_path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

            if "book_id" not in record:
                raise ValueError(f"Missing book_id on line {line_number}")

            # String keys allow matching IDs whether the source used a number
            # or a string.
            book_id = str(record["book_id"])

            if book_id in book_vectors:
                raise ValueError(
                    f"Duplicate book_id {book_id!r} on line {line_number}"
                )

            book_vectors[book_id] = {
                "emo_lex": record.get("emo_lex"),
                "nrc_eil": record.get("nrc_eil"),
            }

    return book_vectors

BOOK_EMOTION_VECTOR_LOOK_UP = load_book_vector_lookup("all_description_emotion_vectors.jsonl")

def get_vector_by_id(book_id, vector_type = "emo_lex"):
    vectors = BOOK_EMOTION_VECTOR_LOOK_UP.get(str(book_id))
    return vectors[vector_type]


print(get_vector_by_id(18329606))