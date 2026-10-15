import json
from tqdm import tqdm
from vector_helper import get_book_vector_by_id, average_vectors, cosine_similarity, bilinear_pool


TEST_DATA_FILE_50_200 = "good_reads_test_data/input/users_profile_test_50_to_200.json"
TEST_DATA_FILE_100_200 = "good_reads_test_data/input/users_profile_test_100_to_200.json"

BIG_TEST_FILE = "Goodreads_data/Organized_users_reviews_filtered_all_above_20_high.json"

TIME_TEST_FILE = "Goodreads_data/users_reviews_all_sorted_by_timestamp.json"

def make_candidate_profile(profile_books, emotion_type, use_bilinear_pool = False):
    emotion_vectors = []
    for book in profile_books:
        book_id = book['book_id']
        try:
            book_vec = get_book_vector_by_id(book_id, emotion_type)
            if use_bilinear_pool:
                emotion_vectors.append(bilinear_pool(book_vec))
            else:
                emotion_vectors.append(book_vec)
        except Exception as e:
            print(e)
    if len(emotion_vectors) == 0:
        raise Exception(f"Missing all profile books")

    emotion_profile = average_vectors(emotion_vectors)

    return emotion_profile

def base_recommender(test_data_file, output_folder, emotion_vector_type, use_bilinear_pool = False):
    with open(test_data_file, 'r') as file:
        data = json.load(file)

    for item in tqdm(data):
        profile_books = item['profile']
        candidate_profile = make_candidate_profile(profile_books, emotion_vector_type, use_bilinear_pool)
        recomendation_books = item['test']
        for book in recomendation_books:
            book_id = book['book_id']
            if use_bilinear_pool:
                temp = get_book_vector_by_id(book_id, emotion_vector_type)
                book_vec = bilinear_pool(temp)
            else: 
                book_vec = get_book_vector_by_id(book_id, emotion_vector_type)
            cos = cosine_similarity(candidate_profile, book_vec)
            book['cos'] = cos

    output_file_path = f"{output_folder}/rec_via_{emotion_vector_type}_and_{'with_bilinear_pool' if use_bilinear_pool else 'no_bilinear_pool'}.json"

    with open(output_file_path, 'w') as f:
        json.dump(data, f, indent=4)

def recommend_with_time(test_data_file, output_folder, emotion_vector_type, use_bilinear_pool = False, min_profile_size = 5, max_profile_size = None):
    with open(test_data_file, 'r') as file:
        data = json.load(file)

    for user in tqdm(data):
        user_profile_books = []
        books = user["reviews"]
        for book in books:
            book_id = book['book_id']
            # If we have enough books to create a profile already, create a recommendation for the book 
            if len(user_profile_books) >= min_profile_size:
                if max_profile_size is None:
                    iter_profile_books = user_profile_books
                else:
                    iter_profile_books = user_profile_books[-max_profile_size:]

                candidate_profile = make_candidate_profile(
                    iter_profile_books,
                    emotion_vector_type,
                    use_bilinear_pool,
                )
                if use_bilinear_pool:
                    temp = get_book_vector_by_id(book_id, emotion_vector_type)
                    book_vec = bilinear_pool(temp)
                else: 
                    book_vec = get_book_vector_by_id(book_id, emotion_vector_type)
                cos = cosine_similarity(candidate_profile, book_vec)
                book['cos'] = cos
            book_rating = book['rating']
            # If the book is highly rated, add the book to user profile books 
            if book_rating >= 4:
                user_profile_books.append(book)

        user["reviews"] = [
            book for book in user["reviews"]
            if "cos" in book
        ]

    max_profile_suffix = (
        f"_max_profile_size_{max_profile_size}" if max_profile_size is not None else ""
    )

    output_file_path = (
        f"{output_folder}/time_rec_via_{emotion_vector_type}_and_"
        f"{'with_bilinear_pool' if use_bilinear_pool else 'no_bilinear_pool'}"
        f"_min_profile_size_{min_profile_size}"
        f"{max_profile_suffix}.json"
    )

    with open(output_file_path, 'w') as f:
            json.dump(data, f, indent=4)

def batch_recommend_with_time(test_data_file, output_folder):
    emotion_vector_types = ("nrc_eil", "emo_lex")
    bilinear_pool_options = (False, True)
    min_profile_sizes = (2, 5, 10)
    max_profile_sizes = (None, 10, 20, 30)

    for emotion_vector_type in emotion_vector_types:
        for use_bilinear_pool in bilinear_pool_options:
            for min_profile_size in min_profile_sizes:
                for max_profile_size in max_profile_sizes:
                    print(f"{emotion_vector_type}, min profile size: {min_profile_size}, max profile size: {max_profile_size}, has bilinear pooling: {use_bilinear_pool}")
                    recommend_with_time(
                        test_data_file=test_data_file,
                        output_folder=output_folder,
                        emotion_vector_type=emotion_vector_type,
                        use_bilinear_pool=use_bilinear_pool,
                        min_profile_size=min_profile_size,
                        max_profile_size=max_profile_size,
                    ) 

def recommend_compare_to_time(test_data_file, output_folder, emotion_vector_type, use_bilinear_pool = False, profile_size = None):
    with open(test_data_file, 'r') as file:
        data = json.load(file)

    for user in tqdm(data):
        has_profile = False
        user_profile = None
        user_profile_books = []
        books = user["reviews"]
        for book in books:
            book_id = book['book_id']
            # If we have enough books to create a profile already, create a recommendation for the book 
            if not has_profile:
                book_rating = book['rating']
                # If the book is highly rated, add the book to user profile books 
                if book_rating >= 4:
                    user_profile_books.append(book)
                if len(user_profile_books) == profile_size:
                    has_profile = True

                    user_profile = make_candidate_profile(
                        user_profile_books,
                        emotion_vector_type,
                        use_bilinear_pool,
                    )
            else:
                if use_bilinear_pool:
                    temp = get_book_vector_by_id(book_id, emotion_vector_type)
                    book_vec = bilinear_pool(temp)
                else: 
                    book_vec = get_book_vector_by_id(book_id, emotion_vector_type)
                cos = cosine_similarity(user_profile, book_vec)
                book['cos'] = cos
            

        user["reviews"] = [
            book for book in user["reviews"]
            if "cos" in book
        ]

    output_file_path = (
        f"{output_folder}/time_compare_rec_via_{emotion_vector_type}_and_"
        f"{'with_bilinear_pool' if use_bilinear_pool else 'no_bilinear_pool'}"
        f"profile_size_{profile_size}.json"
    )

    with open(output_file_path, 'w') as f:
            json.dump(data, f, indent=4)


def batch_recommend_with_time(test_data_file, output_folder):
    emotion_vector_types = ("nrc_eil", "emo_lex")
    bilinear_pool_options = (False, True)
    profile_sizes = (2, 5, 10, 20, 30)
    for emotion_vector_type in emotion_vector_types:
        for use_bilinear_pool in bilinear_pool_options:
            for profile_size in profile_sizes:
                print(f"{emotion_vector_type}, profile size: {profile_size}, has bilinear pooling: {use_bilinear_pool}")
                recommend_compare_to_time(
                    test_data_file=test_data_file,
                    output_folder=output_folder,
                    emotion_vector_type=emotion_vector_type,
                    use_bilinear_pool=use_bilinear_pool,
                    profile_size=profile_size
                ) 

#base_recommender(BIG_TEST_FILE, "good_reads_test_data/results/big_test", "nrc_eil", use_bilinear_pool=False)

#recommend_with_time(TIME_TEST_FILE, "good_reads_test_data/time_results", "nrc_eil", False, 5, 20)

#batch_recommend_with_time(TIME_TEST_FILE, "good_reads_test_data/time_results",)

batch_recommend_with_time(TIME_TEST_FILE, "good_reads_test_data/time_comparison_results")

#recommend_compare_to_time(TIME_TEST_FILE, "good_reads_test_data/time_comparison_results", "nrc_eil", False, 2)