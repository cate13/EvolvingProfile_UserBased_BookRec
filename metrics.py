import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
import csv

def evaluate_recommendations(
    data, k_list=[3, 5, 10], relevance_threshold=4
):
  """Computes metrics only on recommendations that have a non-null

  'predicted_rating'.
  """
  recs = data.get("recommendations", [])
  total_recs = len(recs)

  # Filter out items where predicted_rating is None/null
  filtered_recs = [
      rec for rec in recs if rec.get("predicted_rating") is not None
  ]
  n_filtered = len(filtered_recs)

  if n_filtered == 0:
    return {
        "total_recommendations": total_recs,
        "evaluated_recommendations": 0,
        "mrr": 0.0,
        "precision": {k: None for k in k_list},
        "spearman": np.nan,
        "ndcg": {k: None for k in k_list},
        "warning": "No recommendations found with a non-null predicted_rating.",
    }

  # Extract true ratings and model predicted scores from the filtered subset
  true_ratings = np.array([rec.get("rating", 0) for rec in filtered_recs])
  pred_scores = np.array([rec.get("predicted_rating") for rec in filtered_recs])

  # --- 1. MRR (Mean Reciprocal Rank) ---
  mrr = 0.0
  for rank, r in enumerate(true_ratings, start=1):
    if r >= relevance_threshold:
      mrr = 1.0 / rank
      break

  # --- 2. Precision@K ---
  precision_results = {}
  for k in k_list:
    if n_filtered >= k:
      top_k_ratings = true_ratings[:k]
      relevant_count = np.sum(top_k_ratings >= relevance_threshold)
      precision_results[k] = float(relevant_count / k)
    else:
      precision_results[k] = None

  # --- 3. Spearman Correlation ---
  if (
      n_filtered > 1
      and len(np.unique(true_ratings)) > 1
      and len(np.unique(pred_scores)) > 1
  ):
    spearman_corr, _ = spearmanr(pred_scores, true_ratings)
  else:
    spearman_corr = np.nan

  # --- 4. NDCG@K (using true ratings & actual model predicted scores) ---
  true_ratings_2d = true_ratings.reshape(1, -1)
  pred_scores_2d = pred_scores.reshape(1, -1)

  ndcg_results = {}
  for k in k_list:
    if n_filtered >= k:
      try:
        # Pass predicted scores as relevance/ranking scores, true ratings as gains
        score = ndcg_score(true_ratings_2d, pred_scores_2d, k=k)
        ndcg_results[k] = float(score)
      except Exception:
        ndcg_results[k] = 0.0
    else:
      ndcg_results[k] = None

  return {
      "total_recommendations": total_recs,
      "evaluated_recommendations": n_filtered,
      "mrr": mrr,
      "precision_at_k": precision_results,
      "spearman_correlation": spearman_corr,
      "ndcg_at_k": ndcg_results,
  }

def evaluate_experiment_folder(
    folder_path, k_list=[3, 5, 10], relevance_threshold=4
):
  """Iterates through all JSON files in a folder, evaluates each user,

  and computes overall experimental macro-averaged metrics.
  """
  folder = Path(folder_path)
  json_files = list(folder.glob("*.json"))

  if not json_files:
    return {"warning": f"No JSON files found in directory: {folder_path}"}

  all_mrrs = []
  all_spearmans = []
  precision_accumulators = {k: [] for k in k_list}
  ndcg_accumulators = {k: [] for k in k_list}

  valid_files_count = 0

  for file_path in json_files:
    try:
      with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

      metrics = evaluate_recommendations(
          data, k_list=k_list, relevance_threshold=relevance_threshold
      )

      # Only accumulate if the user file had valid evaluated recommendations
      if metrics["evaluated_recommendations"] > 0:
        valid_files_count += 1
        all_mrrs.append(metrics["mrr"])

        # Handle potential NaN values for Spearman
        sp = metrics["spearman_correlation"]
        if not np.isnan(sp):
          all_spearmans.append(sp)

        # Accumulate P@K for Ks that were long enough for this user
        for k in k_list:
          p_val = metrics["precision_at_k"].get(k)
          if p_val is not None:
            precision_accumulators[k].append(p_val)

          n_val = metrics["ndcg_at_k"].get(k)
          if n_val is not None:
            ndcg_accumulators[k].append(n_val)

    except Exception as e:
      print(f"Error parsing {file_path.name}: {e}")

  # Compute macro-averages across all valid user files
  summary = {
      "total_files_found": len(json_files),
      "valid_files_evaluated": valid_files_count,
      "mrr": float(np.mean(all_mrrs)) if all_mrrs else 0.0,
      "spearman": float(np.mean(all_spearmans)) if all_spearmans else None,
      "precision_at_k": {
          k: float(np.mean(vals)) if vals else None
          for k, vals in precision_accumulators.items()
      },
      "ndcg_at_k": {
          k: float(np.mean(vals)) if vals else None
          for k, vals in ndcg_accumulators.items()
      },
  }

  return summary

def batch_evaluate_parent_folder(
    parent_folder_path,
    output_csv_path="experiment_summary.csv",
    k_list=[3, 5, 10],
    relevance_threshold=4,
):
  """Scans all subfolders in a parent folder, evaluates each experiment,

  and writes the results to a CSV file.
  """
  parent_dir = Path(parent_folder_path)
  subfolders = [f for f in parent_dir.iterdir() if f.is_dir()]

  if not subfolders:
    print(f"No subfolders found in parent directory: {parent_folder_path}")
    return

  csv_rows = []

  for folder in sorted(subfolders):
    print(f"Evaluating run folder: {folder.name}...")
    summary = evaluate_experiment_folder(
        folder, k_list=k_list, relevance_threshold=relevance_threshold
    )

    if summary is None:
      print(f"  -> Skipped (no valid JSON files found).")
      continue

    # Flatten the nested dictionary so it formats cleanly into a single CSV row
    row = {
        "run_folder": folder.name,
        "total_files": summary["total_files_found"],
        "valid_files": summary["valid_files_evaluated"],
        "mean_mrr": summary["mrr"],
        "mean_spearman": summary["spearman"],
    }

    # Flatten precision@k dictionary (e.g., precision_at_3, precision_at_5)
    for k, val in summary["precision_at_k"].items():
      row[f"precision_at_{k}"] = val

    # Flatten ndcg@k dictionary (e.g., ndcg_at_3, ndcg_at_5)
    for k, val in summary["ndcg_at_k"].items():
      row[f"ndcg_at_{k}"] = val

    csv_rows.append(row)

  if not csv_rows:
    print("No valid experiment results found to write.")
    return

  # Write to CSV
  fieldnames = list(csv_rows[0].keys())
  with open(
      output_csv_path, mode="w", newline="", encoding="utf-8"
  ) as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)

  print(f"\nSuccessfully saved summary for {len(csv_rows)} runs to: {output_csv_path}")


batch_evaluate_parent_folder('Goodreads_review_test_results')