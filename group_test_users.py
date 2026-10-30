import csv
import json
from pathlib import Path

from itertools import combinations
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

INPUT_FILES = {
    "Books": Path("Goodreads_review_test_set/nearest_neighbors_by_books.csv"),
    "EmoLex": Path("Goodreads_review_test_set/nearest_neighbors_by_emotion_emo_lex.csv"),
    "NRC-EIL": Path("Goodreads_review_test_set/nearest_neighbors_by_emotion_nrc-eil.csv"),
}

# Directories and files
REVIEW_DIR = Path("Goodreads_review_test_set")
BOOK_VECTORS_FILE = Path(
    "Goodreads_book_vectors/all_description_emotion_vectors.jsonl"
)

NEIGHBOR_COLUMNS = [
    ("1st nearest neighbor", "1st nearest neighbor score"),
    ("2nd nearest neighbor", "2nd nearest neighbor score"),
    ("3rd nearest neighbor", "3rd nearest neighbor score"),
]

OUTPUT_DIR = Path("Goodreads_review_test_set")
COMPARE_OUTPUT_DIR = Path("Goodreads_review_test_set/neighbor_comparison")
COMPARE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BOOK_NEIGHBORS_FILE = OUTPUT_DIR / "nearest_neighbors_by_books.csv"
EMOTION_NEIGHBORS_FILE = OUTPUT_DIR / "nearest_neighbors_by_emotion.csv"

# Emotion representation used for average user profiles.
# Change to "emo_lex" if preferred.
EMOTION_TYPE = "emo_lex"

# Number of neighbors to find
N_NEIGHBORS = 3

# Number of users processed per similarity batch
BATCH_SIZE = 256

CSV_HEADERS = [
    "user_id",
    "1st nearest neighbor",
    "1st nearest neighbor score",
    "2nd nearest neighbor",
    "2nd nearest neighbor score",
    "3rd nearest neighbor",
    "3rd nearest neighbor score",
]


def normalize_book_id(book_id):
    return str(book_id).strip()


def load_book_vectors():
    """Load book-level emotion vectors, indexed by book_id."""
    book_vectors = {}

    with BOOK_VECTORS_FILE.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading book vectors"):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            book_id = normalize_book_id(record["book_id"])
            book_vectors[book_id] = record

    print(f"Loaded vectors for {len(book_vectors):,} books.")
    return book_vectors


def load_user_books():
    """
    Load each user's distinct books from the review test set.

    Returns:
        user_books: dict mapping user_id -> set of book IDs
    """
    user_books = {}

    files = sorted(REVIEW_DIR.glob("*.json"))

    for file_path in tqdm(files, desc="Loading user reviews"):
        with file_path.open("r", encoding="utf-8") as f:
            reviews = json.load(f)

        if not reviews:
            continue

        user_id = str(reviews[0]["user_id"])
        books = {
            normalize_book_id(review["book_id"])
            for review in reviews
        }

        user_books[user_id] = books

    print(f"Loaded {len(user_books):,} users.")
    return user_books


def create_book_matrix(user_ids, user_books):
    """
    Create a sparse binary user-book matrix.

    Each row represents a user and each column represents a book.
    A value of 1 indicates that the user read the book.
    """
    all_books = sorted({
        book_id
        for books in user_books.values()
        for book_id in books
    })

    book_to_index = {
        book_id: index
        for index, book_id in enumerate(all_books)
    }

    rows = []
    cols = []

    for row_index, user_id in enumerate(user_ids):
        for book_id in user_books[user_id]:
            rows.append(row_index)
            cols.append(book_to_index[book_id])

    data = np.ones(len(rows), dtype=np.int32)

    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_ids), len(all_books)),
        dtype=np.int32,
    )

    return matrix


def create_average_emotion_matrix(user_ids, user_books, book_vectors):
    """
    Create one average book-level emotion vector per user.

    For each user:
      1. Retrieve each distinct book's emotion vector.
      2. Skip books without the selected emotion vector.
      3. Average the emotion values across the available books.

    The rating and review-level emotion vectors are not used.
    """
    emotion_keys = [
        "Anger",
        "Anticipation",
        "Disgust",
        "Fear",
        "Joy",
        "Sadness",
        "Surprise",
        "Trust",
    ]

    user_vectors = np.zeros(
        (len(user_ids), len(emotion_keys)),
        dtype=np.float64,
    )

    users_without_vectors = 0

    for row_index, user_id in enumerate(
        tqdm(user_ids, desc="Creating average emotion profiles")
    ):
        vectors = []

        for book_id in user_books[user_id]:
            book_record = book_vectors.get(book_id)

            if book_record is None:
                continue

            emotion_vector = book_record.get(EMOTION_TYPE)

            if not isinstance(emotion_vector, dict):
                continue

            if not all(key in emotion_vector for key in emotion_keys):
                continue

            vectors.append([
                float(emotion_vector[key])
                for key in emotion_keys
            ])

        if vectors:
            user_vectors[row_index] = np.mean(vectors, axis=0)
        else:
            users_without_vectors += 1

    print(
        f"Users without usable {EMOTION_TYPE} vectors: "
        f"{users_without_vectors:,}"
    )

    return user_vectors


def get_top_neighbors(user_ids, similarity_function, output_file):
    """
    Calculate the top three neighbors in batches.

    similarity_function(start, end) must return similarities between
    users[start:end] and all users.

    Self-similarity is excluded. Users are ranked by descending score.
    """
    n_users = len(user_ids)

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)

        for start in tqdm(
            range(0, n_users, BATCH_SIZE),
            desc=f"Finding neighbors: {output_file.name}",
        ):
            end = min(start + BATCH_SIZE, n_users)

            similarities = similarity_function(start, end)

            # A user cannot be their own neighbor.
            for local_index, global_index in enumerate(range(start, end)):
                similarities[local_index, global_index] = -np.inf

            for local_index, global_index in enumerate(range(start, end)):
                scores = similarities[local_index]

                # Exclude users with no finite similarity.
                valid_indices = np.flatnonzero(np.isfinite(scores))

                if len(valid_indices) == 0:
                    writer.writerow([user_ids[global_index], "", "", "", "", "", ""])
                    continue

                top_indices = valid_indices[
                    np.argsort(scores[valid_indices])[::-1][:N_NEIGHBORS]
                ]

                row = [user_ids[global_index]]

                for neighbor_index in top_indices:
                    row.append(user_ids[neighbor_index])
                    row.append(round(float(scores[neighbor_index]), 6))

                # Fill any missing neighbor columns if fewer than 3 users
                # other than the current user are available.
                while len(row) < len(CSV_HEADERS):
                    row.extend(["", ""])

                writer.writerow(row[:len(CSV_HEADERS)])

    print(f"Saved: {output_file}")


def group_main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the input data.
    user_books = load_user_books()
    book_vectors = load_book_vectors()

    # Consistent ordering across both outputs.
    user_ids = sorted(user_books.keys())

    if len(user_ids) < 2:
        print("At least two users are required to find neighbors.")
        return

    # ---------------------------------------------------------
    # Method 1: Similarity based on books read (Jaccard)
    # ---------------------------------------------------------
    print("\nCreating user-book matrix...")

    book_matrix = create_book_matrix(user_ids, user_books)

    # Number of distinct books read by each user.
    book_counts = np.asarray(
        book_matrix.sum(axis=1)
    ).ravel()

    def book_similarity_batch(start, end):
        # Number of books shared by each pair of users.
        intersections = (
            book_matrix[start:end] @ book_matrix.T
        ).toarray().astype(np.float64)

        # Jaccard = intersection / union.
        unions = (
            book_counts[start:end, None]
            + book_counts[None, :]
            - intersections
        )

        similarities = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections),
            where=unions > 0,
        )

        return similarities

    print("\nFinding neighbors based on books read...")

    get_top_neighbors(
        user_ids=user_ids,
        similarity_function=book_similarity_batch,
        output_file=BOOK_NEIGHBORS_FILE,
    )

    # ---------------------------------------------------------
    # Method 2: Similarity based on average emotion vectors
    # ---------------------------------------------------------
    print("\nCreating average emotion vectors...")

    emotion_matrix = create_average_emotion_matrix(
        user_ids=user_ids,
        user_books=user_books,
        book_vectors=book_vectors,
    )

    def emotion_similarity_batch(start, end):
        return cosine_similarity(
            emotion_matrix[start:end],
            emotion_matrix,
        )

    print("\nFinding neighbors based on average emotion vectors...")

    get_top_neighbors(
        user_ids=user_ids,
        similarity_function=emotion_similarity_batch,
        output_file=EMOTION_NEIGHBORS_FILE,
    )

    print("\nAll done!")
    print(f"Book-based neighbors: {BOOK_NEIGHBORS_FILE}")
    print(f"Emotion-based neighbors: {EMOTION_NEIGHBORS_FILE}")

def load_neighbors(file_path):
    """Load each user's ordered list of neighbors and their scores."""
    neighbors = {}

    with file_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            user_id = row["user_id"].strip()
            entries = []

            for neighbor_col, score_col in NEIGHBOR_COLUMNS:
                neighbor_id = (row.get(neighbor_col) or "").strip()
                score_text = (row.get(score_col) or "").strip()

                if not neighbor_id:
                    continue

                score = float(score_text) if score_text else None
                entries.append({
                    "user_id": neighbor_id,
                    "score": score,
                })

            neighbors[user_id] = entries

    return neighbors

def compare_user_neighbors(neighbors_a, neighbors_b):
    """
    Compare two methods for users present in both files.

    Metrics:
      - overlap_count: number of shared neighbors
      - overlap_fraction: shared neighbors / union of neighbors
      - rank agreement: Spearman correlation on shared neighbors,
        provided at least two neighbors are shared
      - exact_order_match: whether the ordered lists are identical
    """
    common_users = sorted(set(neighbors_a) & set(neighbors_b))
    results = []

    for user_id in tqdm(common_users, desc="Comparing users", leave=False):
        list_a = neighbors_a[user_id]
        list_b = neighbors_b[user_id]

        ids_a = [entry["user_id"] for entry in list_a]
        ids_b = [entry["user_id"] for entry in list_b]

        set_a = set(ids_a)
        set_b = set(ids_b)

        shared = set_a & set_b
        union = set_a | set_b

        overlap_count = len(shared)
        overlap_fraction = (
            overlap_count / len(union) if union else 1.0
        )

        # Rank differences for neighbors appearing in both lists.
        # A rank of 1 means first nearest neighbor.
        ranks_a = {neighbor_id: rank for rank, neighbor_id in enumerate(ids_a, 1)}
        ranks_b = {neighbor_id: rank for rank, neighbor_id in enumerate(ids_b, 1)}

        rank_differences = [
            abs(ranks_a[neighbor_id] - ranks_b[neighbor_id])
            for neighbor_id in shared
        ]

        mean_rank_difference = (
            float(np.mean(rank_differences)) if rank_differences else None
        )

        # Spearman rank correlation is meaningful for at least two
        # shared neighbors, and is calculated using their relative ranks.
        spearman = None
        if len(shared) >= 2:
            shared_ids = sorted(shared)
            rank_values_a = [ranks_a[n] for n in shared_ids]
            rank_values_b = [ranks_b[n] for n in shared_ids]

            # Correlation is undefined if either rank list is constant.
            if (
                np.std(rank_values_a) > 0
                and np.std(rank_values_b) > 0
            ):
                spearman = float(
                    np.corrcoef(rank_values_a, rank_values_b)[0, 1]
                )

        results.append({
            "user_id": user_id,
            "neighbors_a": ids_a,
            "neighbors_b": ids_b,
            "shared_neighbors": sorted(shared),
            "overlap_count": overlap_count,
            "overlap_fraction": overlap_fraction,
            "mean_rank_difference": mean_rank_difference,
            "spearman_rank_correlation": spearman,
            "exact_order_match": ids_a == ids_b,
        })

    return results

def write_detailed_results(results, method_a, method_b, output_file):
    """Write per-user comparisons to CSV."""
    headers = [
        "user_id",
        f"{method_a} neighbors",
        f"{method_b} neighbors",
        "shared neighbors",
        "overlap count",
        "overlap fraction",
        "mean rank difference",
        "spearman rank correlation",
        "exact order match",
    ]

    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for result in results:
            writer.writerow({
                "user_id": result["user_id"],
                f"{method_a} neighbors": " | ".join(result["neighbors_a"]),
                f"{method_b} neighbors": " | ".join(result["neighbors_b"]),
                "shared neighbors": " | ".join(result["shared_neighbors"]),
                "overlap count": result["overlap_count"],
                "overlap fraction": round(result["overlap_fraction"], 6),
                "mean rank difference": (
                    round(result["mean_rank_difference"], 6)
                    if result["mean_rank_difference"] is not None else ""
                ),
                "spearman rank correlation": (
                    round(result["spearman_rank_correlation"], 6)
                    if result["spearman_rank_correlation"] is not None else ""
                ),
                "exact order match": result["exact_order_match"],
            })

def summarize_comparison(results, method_a, method_b):
    """Calculate aggregate similarity statistics."""
    if not results:
        return None

    overlap_counts = [r["overlap_count"] for r in results]
    overlap_fractions = [r["overlap_fraction"] for r in results]
    exact_matches = [r["exact_order_match"] for r in results]

    rank_differences = [
        r["mean_rank_difference"]
        for r in results
        if r["mean_rank_difference"] is not None
    ]

    spearman_values = [
        r["spearman_rank_correlation"]
        for r in results
        if r["spearman_rank_correlation"] is not None
    ]

    return {
        "method_a": method_a,
        "method_b": method_b,
        "users_compared": len(results),
        "average_shared_neighbors": round(float(np.mean(overlap_counts)), 6),
        "average_overlap_fraction": round(float(np.mean(overlap_fractions)), 6),
        "users_with_all_3_neighbors_shared": sum(
            count == 3 for count in overlap_counts
        ),
        "percent_users_with_all_3_neighbors_shared": round(
            100 * sum(count == 3 for count in overlap_counts) / len(results), 4
        ),
        "exact_order_matches": sum(exact_matches),
        "percent_exact_order_matches": round(
            100 * sum(exact_matches) / len(results), 4
        ),
        "average_rank_difference_shared_neighbors": (
            round(float(np.mean(rank_differences)), 6)
            if rank_differences else ""
        ),
        "average_spearman_rank_correlation": (
            round(float(np.mean(spearman_values)), 6)
            if spearman_values else ""
        ),
        "users_with_spearman_defined": len(spearman_values),
    }


def main():
    all_neighbors = {}

    for method, file_path in INPUT_FILES.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        all_neighbors[method] = load_neighbors(file_path)
        print(f"{method}: loaded {len(all_neighbors[method]):,} users")

    summaries = []

    for method_a, method_b in combinations(INPUT_FILES.keys(), 2):
        print(f"\nComparing {method_a} vs {method_b}...")

        results = compare_user_neighbors(
            all_neighbors[method_a],
            all_neighbors[method_b],
        )

        if not results:
            print("No common users found.")
            continue

        detailed_file = COMPARE_OUTPUT_DIR / (
            f"comparison_{method_a.lower()}_vs_{method_b.lower()}.csv"
        )

        write_detailed_results(
            results,
            method_a,
            method_b,
            detailed_file,
        )

        summary = summarize_comparison(results, method_a, method_b)
        summaries.append(summary)

        print(f"Users compared: {summary['users_compared']:,}")
        print(
            f"Average shared neighbors: "
            f"{summary['average_shared_neighbors']:.3f} / 3"
        )
        print(
            f"Average overlap fraction: "
            f"{summary['average_overlap_fraction']:.3f}"
        )
        print(
            f"Exact same ordered list: "
            f"{summary['percent_exact_order_matches']:.2f}%"
        )
        print(f"Detailed results: {detailed_file}")

    summary_file = COMPARE_OUTPUT_DIR / "neighbor_comparison_summary.csv"

    if summaries:
        with summary_file.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(summaries[0].keys()),
            )
            writer.writeheader()
            writer.writerows(summaries)

        print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()