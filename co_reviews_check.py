
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from tqdm import tqdm


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_DIR = Path("Goodreads_data/1st10000users")
OUTPUT_FILE = Path("Goodreads_data/overlapping_user_pairs_10.csv")

MIN_SHARED_BOOKS = 10


def load_user_books():
    """Load each user's distinct book IDs from all JSON files."""

    user_books = {}

    json_files = sorted(INPUT_DIR.glob("*.json"))

    print(f"Found {len(json_files)} JSON files.")

    for file_path in tqdm(json_files, desc="Loading users"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reviews = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"\nSkipping {file_path.name}: {e}")
            continue

        # The filename is the user ID if the file contains no reviews.
        user_id = file_path.stem

        # Use the user_id stored in the reviews when available.
        if reviews and reviews[0].get("user_id") is not None:
            user_id = str(reviews[0]["user_id"])

        # Use a set so duplicate reviews of a book count only once.
        books = {
            str(review["book_id"])
            for review in reviews
            if review.get("book_id") is not None
        }

        user_books[user_id] = books

    return user_books


def build_book_index(user_books):
    """Map each book ID to the users who reviewed it."""

    book_users = defaultdict(list)

    for user_id, books in tqdm(
        user_books.items(), desc="Building book index"
    ):
        for book_id in books:
            book_users[book_id].append(user_id)

    return book_users


def find_overlapping_pairs(user_books, book_users):
    """
    Count shared distinct books for each user pair.

    Each book contributes at most one count to a pair because
    user_books contains sets of distinct book IDs.
    """

    pair_counts = Counter()

    for users in tqdm(
        book_users.values(),
        desc="Counting shared books",
    ):
        # Each pair of users reviewing this book shares one book.
        for user_a, user_b in combinations(sorted(users), 2):
            pair_counts[(user_a, user_b)] += 1

    # Keep only pairs meeting the minimum overlap threshold.
    qualifying_pairs = []

    for (user_a, user_b), shared_count in pair_counts.items():
        if shared_count < MIN_SHARED_BOOKS:
            continue

        total_distinct_books = (
            len(user_books[user_a])
            + len(user_books[user_b])
            - shared_count
        )

        jaccard = (
            shared_count / total_distinct_books
            if total_distinct_books
            else 0.0
        )

        qualifying_pairs.append({
            "user_id_1": user_a,
            "user_id_2": user_b,
            "shared_book_count": shared_count,
            "user_1_book_count": len(user_books[user_a]),
            "user_2_book_count": len(user_books[user_b]),
            "jaccard_similarity": jaccard,
        })

    qualifying_pairs.sort(
        key=lambda pair: (
            -pair["shared_book_count"],
            -pair["jaccard_similarity"],
        )
    )

    return qualifying_pairs


def save_results(pairs):
    """Save qualifying user pairs to a CSV file."""

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "user_id_1",
        "user_id_2",
        "shared_book_count",
        "user_1_book_count",
        "user_2_book_count",
        "jaccard_similarity",
    ]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)


def main():
    user_books = load_user_books()

    print(f"\nLoaded {len(user_books)} users.")
    print(
        f"Total distinct user-book interactions: "
        f"{sum(len(books) for books in user_books.values()):,}"
    )

    book_users = build_book_index(user_books)

    print(f"\nDistinct books indexed: {len(book_users):,}")

    qualifying_pairs = find_overlapping_pairs(
        user_books,
        book_users,
    )

    save_results(qualifying_pairs)

    paired_users = set()

    for pair in qualifying_pairs:
        paired_users.add(pair["user_id_1"])
        paired_users.add(pair["user_id_2"])

    print("\nFinished!")
    print(f"Minimum shared books: {MIN_SHARED_BOOKS}")
    print(f"Qualifying user pairs: {len(qualifying_pairs):,}")
    print(f"Unique users with at least one qualifying pair: {len(paired_users):,}")
    print(f"Results saved to: {OUTPUT_FILE}")

    if qualifying_pairs:
        print("\nTop 10 pairs by shared-book count:")

        for pair in qualifying_pairs[:10]:
            print(
                f"{pair['user_id_1']} <-> {pair['user_id_2']} | "
                f"Shared books: {pair['shared_book_count']} | "
                f"Jaccard: {pair['jaccard_similarity']:.4f}"
            )


if __name__ == "__main__":
    main()
