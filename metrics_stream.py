import json
import os
import csv
import math
import random
import numpy as np
from sklearn.metrics import ndcg_score
import scipy.stats as stats
from scipy.stats import wilcoxon
import re
from tqdm import tqdm 
import pandas as pd
from itertools import zip_longest

RELEVANT_RATING = 4

def relevant(rating):
    if rating >= RELEVANT_RATING:
        return True 
    return False

def get_rr(book_list):
    x = 1
    for book in book_list:
        if relevant(book):
            return 1.0/x
        x+=1
    return 0.0

def precision_at_k(ranked_ratings, k):
    if k <= 0:
        return 0.0
    top_k_items = ranked_ratings[:k]
    relevant_count = sum(1 for item in top_k_items if relevant(item))
    return relevant_count / k


def handle_user(user, json_index = "test"):
    #print(user['id'])
    recommendations = user[json_index]
    filtered = [rec for rec in recommendations if rec["rating"] != 0]
    # The filtering should be redundent but just in case 

    recommendations_sorted = sorted(filtered, key=lambda r: r["cos"], reverse=True)
    rating_for_recommendations_sorted = [rec["rating"] for rec in recommendations_sorted]

    rr = get_rr(rating_for_recommendations_sorted)
    p3 = p5 = None
    p1 = precision_at_k(rating_for_recommendations_sorted, k=1)
    if len(rating_for_recommendations_sorted) >= 3:
        p3 = precision_at_k(rating_for_recommendations_sorted, k=3)
        if len(rating_for_recommendations_sorted) >= 5:
            p5 = precision_at_k(rating_for_recommendations_sorted, k=5)

    scores = [rec["cos"] for rec in filtered]
    ratings = [rec["rating"] for rec in filtered]

    scores = np.array([scores])
    ratings = np.array([ratings])

    scores = [float(rec["cos"]) for rec in filtered]
    ratings = [int(rec["rating"]) for rec in filtered]

    rho = None
    # Guard check: Ensure we have at least 2 items to compute ranking metrics
    if len(scores) > 1:
        y_score = np.array([scores]) 
        y_true = np.array([ratings])

        ndcg_2 = ndcg_score(y_true, y_score, k=2)
        ndcg_3 = ndcg_5 = ndcg_10 = None
        if len(scores) >= 3:
            ndcg_3 = ndcg_score(y_true, y_score, k=3)
            if len(scores) >= 5:
                ndcg_5 = ndcg_score(y_true, y_score, k=5)
                if len(scores) >= 10:
                    ndcg_10 = ndcg_score(y_true, y_score, k=10)

        # Ensure both arrays have variation before calling spearmanr
        if len(set(scores)) > 1 and len(set(ratings)) > 1:
            rho, _ = stats.spearmanr(scores, ratings)
    else:
        ndcg_2 = ndcg_3 = ndcg_5 = ndcg_10 = None


    return rr, p1, p3, p5, (abs(rho) if rho is not None else None), ndcg_2, ndcg_3, ndcg_5, ndcg_10

def evaluate_json_object(records, json_index = "reviews"):

    rr_list = []
    p1_list = []
    p3_list = []
    p5_list = []
    spearman_list = []
    ndcg_2_list = []
    ndcg_3_list = []
    ndcg_5_list = []
    ndcg_10_list = []

    for record in records:
        rr, p1, p3, p5, rho, ndcg_2, ndcg_3, ndcg_5, ndcg_10 = handle_user(record, json_index)

        rr_list.append(rr)
        p1_list.append(p1)
        p3_list.append(p3)
        p5_list.append(p5)

        spearman_list.append(rho)
        ndcg_2_list.append(ndcg_2)
        ndcg_3_list.append(ndcg_3)
        ndcg_5_list.append(ndcg_5)
        ndcg_10_list.append(ndcg_10)

    return {
        "mrr" : rr_list,
        "P@1" : p1_list,
        "P@3" : p3_list,
        "P@5" : p5_list,        
        "Spearman" : spearman_list,
        "NDCG_2" : ndcg_2_list,
        "NDCG_3" : ndcg_3_list,
        "NDCG_5" : ndcg_5_list,
        "NDCG_10" : ndcg_10_list,
    }

METRIC_NAMES = [
    "mrr",
    "P@1",
    "P@3",
    "P@5",
    "Spearman",
    "NDCG_2",
    "NDCG_3",
    "NDCG_5",
    "NDCG_10",
]

def calculate_metrics_by_user(records, json_index="reviews"):
    metrics_by_user = {}

    for record in records:
        metric_values = handle_user(record, json_index)

        metrics_by_user[record["id"]] = dict(
            zip(METRIC_NAMES, metric_values)
        )

    return metrics_by_user

def calculate_metrics_from_object(json_object):
    metrics_by_user = calculate_metrics_by_user(
        json_object,
    )

    overall_metrics = {}

    for metric in METRIC_NAMES:
        values = [
            user_metrics[metric]
            for user_metrics in metrics_by_user.values()
            if user_metrics[metric] is not None
            and not math.isnan(user_metrics[metric])
        ]

        overall_metrics[metric] = (
            np.mean(values) if values else None
        )

    return overall_metrics

def formate_file_name(filename):
    name = os.path.splitext(filename)[0]

    # Remove common prefix
    name = name.replace("rec_via_", "")

    # Determine bilinear pooling
    if "_with_bilinear_pool" in name:
        bilinear = "with Bilinear Pool"
        name = name.replace("_with_bilinear_pool", "")
    elif "_no_bilinear_pool" in name:
        bilinear = "without Bilinear Pool"
        name = name.replace("_no_bilinear_pool", "")
    else:
        bilinear = ""

    # Rename emotion methods
    name = name.replace("nrc_eil", "NRC-EIL")
    name = name.replace("emo_lex", "EmoLex")

    # Handle any additional methods you might have
    name = name.replace("_and_", " ")
    name = name.replace("_", " ")

    # Capitalize / clean up
    name = name.strip()

    if bilinear:
        name = f"{name} {bilinear}"

    return name

def extract_filename_info(filename):
    pattern = re.compile(
        r"^time_rec_via_(?P<emotion_type>.+?)"
        r"_and_(?P<pooling>no|with)_bilinear_pool"
        r"_min_profile_size_(?P<min_size>\d+)"
        r"(?:_max_profile_size_(?P<max_size>\d+))?"
        r"\.json$"
    )

    match = pattern.match(filename)

    if not match:
        return None  # Filename does not match the expected format

    info = match.groupdict()

    return {
        "emotion_type": info["emotion_type"],
        "use_bilinear_pool": info["pooling"] == "with",
        "min_profile_size": int(info["min_size"]),
        "max_profile_size": (
            int(info["max_size"])
            if info["max_size"] is not None
            else None
        ),
    }

def extract_compare_filename_info(filename):
    pattern = re.compile(
        r"^time_compare_rec_via_(?P<emotion_type>.+?)"
        r"_and_(?P<pooling>no|with)_bilinear_pool"
        r"profile_size_(?P<profile_size>\d+)"
        r"\.json$"
    )

    match = pattern.match(filename)

    if not match:
        return None

    info = match.groupdict()

    return {
        "emotion_type": info["emotion_type"],
        "use_bilinear_pool": info["pooling"] == "with",
        "profile_size": int(info["profile_size"]),
    }

def filter_shared_reviews(data1, data2, json_index="reviews"):
    data1_by_id = {record["id"]: record for record in data1}
    data2_by_id = {record["id"]: record for record in data2}

    shared_user_ids = sorted(
        set(data1_by_id.keys()) & set(data2_by_id.keys())
    )

    filtered_data1 = []
    filtered_data2 = []

    for user_id in shared_user_ids:
        record1 = data1_by_id[user_id]
        record2 = data2_by_id[user_id]

        book_ids1 = {
            review["book_id"]
            for review in record1[json_index]
        }

        book_ids2 = {
            review["book_id"]
            for review in record2[json_index]
        }

        shared_book_ids = book_ids1 & book_ids2

        # Skip users who have no shared reviews
        if not shared_book_ids:
            continue

        # Copy the user objects so the originals are not modified
        filtered_record1 = dict(record1)
        filtered_record2 = dict(record2)

        filtered_record1[json_index] = [
            review
            for review in record1[json_index]
            if review["book_id"] in shared_book_ids
        ]

        filtered_record2[json_index] = [
            review
            for review in record2[json_index]
            if review["book_id"] in shared_book_ids
        ]

        filtered_data1.append(filtered_record1)
        filtered_data2.append(filtered_record2)

    return filtered_data1, filtered_data2

def write_rows_to_csv(rows, output_path):
    # Preserve the order in which fields first appear
    fieldnames = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        if fieldnames:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames
            )
            writer.writeheader()
            writer.writerows(rows)

    print(f"Results written to {output_path}")

def control_for_recomendation_length(target_folder, baseline_filename, json_index="reviews", csv_file_name = "controlled_wilcoxon.csv", wilcoxon_csv_file_name="controlled_wilcoxon.csv"):
    baseline_path = os.path.join(target_folder, baseline_filename)

    all_files = [f for f in os.listdir(target_folder) 
                if f.endswith('.json') and f != baseline_filename]

    results_rows_wilcoxon = []
    result_rows = []

    for file in all_files:
        comp_path = os.path.join(target_folder, file)
        with open(comp_path, "r", encoding="utf-8") as f:
            time_records = json.load(f)

        with open(baseline_path, "r", encoding="utf-8") as f:
            base_records = json.load(f)

        filtered_time_records, filtered_base_records = (
            filter_shared_reviews(
                time_records,
                base_records,
                json_index=json_index
            )
        )

        # Handle files with no usable shared users
        if not filtered_time_records:
            print(
                f"Skipping {file}: "
                "no users with shared reviews."
            )
            continue

        # Calculate metrics by user ID
        time_metrics_by_user = calculate_metrics_by_user(
            filtered_time_records,
        )

        base_metrics_by_user = calculate_metrics_by_user(
            filtered_base_records,
        )

        time_metrics = calculate_metrics_from_object(
            filtered_time_records,
        )

        run_details = extract_filename_info(file)

        result_rows.append({
            "Emotion Vector Type": run_details["emotion_type"],
            "Used Bilinear Pooling": run_details["use_bilinear_pool"],
            "Min Profile Size": run_details["min_profile_size"],
            "Max Profile Size": run_details["max_profile_size"],
            "mrr": time_metrics["mrr"],
            "P@1": time_metrics["P@1"],
            "P@3": time_metrics["P@3"],
            "P@5": time_metrics["P@5"],
            "Spearman": time_metrics["Spearman"],
            "NDCG_2": time_metrics["NDCG_2"],
            "NDCG_3": time_metrics["NDCG_3"],
            "NDCG_5": time_metrics["NDCG_5"],
            "NDCG_10": time_metrics["NDCG_10"],
        })

        shared_user_ids = sorted(
            set(base_metrics_by_user.keys())
            & set(time_metrics_by_user.keys())
        )
        
        # Wilcoxon 
        for metric in METRIC_NAMES:
            base_scores = []
            comp_scores = []

            for user_id in shared_user_ids:
                base_value = base_metrics_by_user[user_id][metric]
                comp_value = time_metrics_by_user[user_id][metric]

                if is_valid(base_value) and is_valid(comp_value):
                    base_scores.append(float(base_value))
                    comp_scores.append(float(comp_value))

            run_details[f"{metric}_n"] = len(base_scores)

            if len(base_scores) < 2:
                run_details[f"{metric}_p_val"] = None
                continue

            base_array = np.array(base_scores)
            comp_array = np.array(comp_scores)

            differences = base_array - comp_array

            if np.all(differences == 0):
                p_val = 1.0
            else:
                _, p_val = stats.wilcoxon(
                    base_array,
                    comp_array,
                    alternative="less"
                )

            run_details[f"{metric}_p_val"] = format_p_value(p_val)

        results_rows_wilcoxon.append(run_details)

    comparison_output_path = os.path.join(
        target_folder,
        csv_file_name
    )

    wilcoxon_output_path = os.path.join(
        target_folder,
        wilcoxon_csv_file_name
    )

    write_rows_to_csv(
        result_rows,
        comparison_output_path
    )

    write_rows_to_csv(
        results_rows_wilcoxon,
        wilcoxon_output_path
    )

def is_valid(value):
    if value is None:
        return False

    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False

def format_p_value(p_value):
    return (
        "{:.2e}".format(p_value)
        if p_value < 0.0001
        else round(float(p_value), 5)
    )


def control_time_wilcoxon_batch():
    input_lists = [
        [f"good_reads_test_data/Time Wilcoxon/{10}_emolex_bilinear_pool", "time_compare_rec_via_emo_lex_and_with_bilinear_poolprofile_size_10.json", "control_emolex_with_bilinear_pool_10.csv", "control_wilcoxon_emolex_with_bilinear_pool_10.csv"],
        [f"good_reads_test_data/Time Wilcoxon/{20}_emolex_bilinear_pool", "time_compare_rec_via_emo_lex_and_with_bilinear_poolprofile_size_20.json", "control_emolex_with_bilinear_pool_20.csv", "control_wilcoxon_emolex_with_bilinear_pool_20.csv"],
        [f"good_reads_test_data/Time Wilcoxon/{30}_emolex_bilinear_pool", "time_compare_rec_via_emo_lex_and_with_bilinear_poolprofile_size_30.json", "control_emolex_with_bilinear_pool_30.csv", "control_wilcoxon_emolex_with_bilinear_pool_30.csv"],
        [f"good_reads_test_data/Time Wilcoxon/{10}_emolex_no_bilinear_pool", "time_compare_rec_via_emo_lex_and_no_bilinear_poolprofile_size_10.json", "control_emolex_no_bilinear_pool_10.csv", "control_wilcoxon_emolex_no_bilinear_pool_10.csv"],
        [f"good_reads_test_data/Time Wilcoxon/{20}_emolex_no_bilinear_pool", "time_compare_rec_via_emo_lex_and_no_bilinear_poolprofile_size_20.json", "control_emolex_no_bilinear_pool_20.csv", "control_wilcoxon_emolex_no_bilinear_pool_20.csv"],
        [f"good_reads_test_data/Time Wilcoxon/{30}_emolex_no_bilinear_pool", "time_compare_rec_via_emo_lex_and_no_bilinear_poolprofile_size_30.json", "control_emolex_no_bilinear_pool_30.csv", "control_wilcoxon_emolex_no_bilinear_pool_30.csv"],
        [f"good_reads_test_data/Time Wilcoxon/{10}_nrc_eil_bilinear_pool", "time_compare_rec_via_nrc_eil_and_with_bilinear_poolprofile_size_10.json", "control_nrc-eil_with_bilinear_pool_10.csv", "control_wilcoxon_nrc-eil_with_bilinear_pool_10.csv"],
        [f"good_reads_test_data/Time Wilcoxon/{20}_nrc_eil_bilinear_pool", "time_compare_rec_via_nrc_eil_and_with_bilinear_poolprofile_size_20.json", "control_nrc-eil_with_bilinear_pool_20.csv", "control_wilcoxon_nrc-eil_with_bilinear_pool_20.csv"],
        [f"good_reads_test_data/Time Wilcoxon/{30}_nrc_eil_bilinear_pool", "time_compare_rec_via_nrc_eil_and_with_bilinear_poolprofile_size_30.json", "control_nrc-eil_with_bilinear_pool_30.csv", "control_wilcoxon_nrc-eil_with_bilinear_pool_30.csv"],
        [f"good_reads_test_data/Time Wilcoxon/{10}_nrc_eil_no_bilinear_pool", "time_compare_rec_via_nrc_eil_and_no_bilinear_poolprofile_size_10.json", "control_nrc-eil_no_bilinear_pool_10.csv", "control_wilcoxon_nrc-eil_no_bilinear_pool_10.csv"],
        [f"good_reads_test_data/Time Wilcoxon/{20}_nrc_eil_no_bilinear_pool", "time_compare_rec_via_nrc_eil_and_no_bilinear_poolprofile_size_20.json", "control_nrc-eil_no_bilinear_pool_20.csv", "control_wilcoxon_nrc-eil_no_bilinear_pool_20.csv"],
        [f"good_reads_test_data/Time Wilcoxon/{30}_nrc_eil_no_bilinear_pool", "time_compare_rec_via_nrc_eil_and_no_bilinear_poolprofile_size_30.json", "control_nrc-eil_no_bilinear_pool_30.csv", "control_wilcoxon_nrc-eil_no_bilinear_pool_30.csv"],
    ]

    for batch in tqdm(input_lists):
        control_for_recomendation_length(
                target_folder=batch[0], 
                baseline_filename=batch[1],
                csv_file_name=batch[2],
                wilcoxon_csv_file_name=batch[3])


control_time_wilcoxon_batch()