# eval_embeddings_ranking.py
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, wilcoxon
from sentence_transformers import SentenceTransformer
from sklearn.metrics import ndcg_score

# ---------------------------
# Config
# ---------------------------
MODEL_BASE = "intfloat/multilingual-e5-small"  # original
MODEL_FINE = (
    "fine_tune/output/multilingual-e5-small-finetuned-using-simcse"  # your finetuned
)
DEVICE = "cuda"  # or "cpu" if you prefer; will auto-fallback if unavailable
BATCH_SIZE = 64  # for encode() batching; reduce if OOM
SAVE_CSV = "dataset_collection/data/eval_results.csv"
NDCG_KS = [1, 3, 5]  # ks to report

# ---------------------------
# Dataset: paste or load your list here
# ---------------------------
# Option A: if you saved dataset to file 'dataset.json':
with open("dataset_collection/data/evaluation_data.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Option B: paste your dataset variable directly below. For example:
# dataset = [
#     # ... paste the JSON objects you provided ...
# ]


# ---------------------------
# Helpers
# ---------------------------
def make_candidate_lists(item: Dict):
    """
    Return:
      - query_text (str)
      - candidates (List[str])
      - gt_ranks (List[int])   # 1-based rank from dataset (1 is best)
    """
    q = item["question"]
    cands = []
    ranks = []
    for c in item.get("relevant_questions", []):
        cands.append(c["content"])
        ranks.append(int(c["rank"]))
    return q, cands, ranks


def compute_rel_gain_from_rank(gt_ranks: List[int]):
    # Convert ground-truth rank (1 best) -> relevance score (higher better)
    # Using rel = max_rank - rank + 1 so that best (rank=1) -> highest relevance.
    if len(gt_ranks) == 0:
        return []
    max_rank = max(gt_ranks)
    rels = [max_rank - r + 1 for r in gt_ranks]
    return rels


def safe_ndcg(y_true_rels: List[int], pred_scores: List[float], k: int):
    if len(y_true_rels) == 0:
        return np.nan
    # ndcg_score expects arrays shape (n_samples, n_labels)
    return ndcg_score([y_true_rels], [pred_scores], k=k)


# ---------------------------
# Load models
# ---------------------------
def load_model(model_path_or_name: str, device: str):
    print("Loading", model_path_or_name, "on", device)
    return SentenceTransformer(model_path_or_name, device=device)


def encode_batch(
    model: SentenceTransformer, texts: List[str], batch_size: int, device: str
):
    # convert_to_numpy -> good for CPU/GPU interchange
    emb = model.encode(
        texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
    )
    return emb


# ---------------------------
# Main evaluation function
# ---------------------------
def evaluate_model_on_dataset(
    model: SentenceTransformer, dataset: List[Dict], batch_size: int
):
    per_query_metrics = []
    for item in dataset:
        q_text, cands, gt_ranks = make_candidate_lists(item)
        if len(cands) == 0:
            # no candidates -> skip or record NaNs
            per_query_metrics.append(
                {
                    "id": item.get("id"),
                    "ndcg@1": np.nan,
                    "ndcg@3": np.nan,
                    "ndcg@5": np.nan,
                    "spearman": np.nan,
                    "kendall": np.nan,
                    "mrr": np.nan,
                    "n_candidates": 0,
                }
            )
            continue

        # encode query + candidates
        texts = [q_text] + cands
        embs = encode_batch(model, texts, batch_size=batch_size, device=model.device)
        q_emb = embs[0]
        cand_embs = embs[1:]

        # cosine similarities
        # normalize
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-12)
        cand_norms = cand_embs / (
            np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-12
        )
        sims = (cand_norms @ q_norm).tolist()

        # predicted rank order (1 is top)
        pred_order = np.argsort(
            -np.array(sims)
        )  # indices of candidates sorted descending scores
        pred_ranks = np.empty(len(cands), dtype=int)
        # convert to 1-based rank position
        for pos, idx in enumerate(pred_order, start=1):
            pred_ranks[idx] = pos

        # ground truth relevance scores for NDCG
        rels = compute_rel_gain_from_rank(gt_ranks)  # higher better

        # compute metrics
        ndcg_vals = {}
        for k in NDCG_KS:
            ndcg_vals[f"ndcg@{k}"] = safe_ndcg(rels, sims, k=k)

        # Spearman/Kendall: compare predicted ranks vs ground truth ranks
        # note: ground truth ranks are 1 is best; convert to arrays of same length
        try:
            sp = spearmanr(gt_ranks, pred_ranks).correlation
        except Exception:
            sp = np.nan
        try:
            kd = kendalltau(gt_ranks, pred_ranks).correlation
        except Exception:
            kd = np.nan

        # MRR: position of the ground-truth rank==1 item in predicted list
        # find index(es) where gt_rank==1 (usually single). compute 1 / position_of_first_best
        best_idxs = [i for i, r in enumerate(gt_ranks) if r == 1]
        if len(best_idxs) == 0:
            mrr = 0.0
        else:
            # take first best if multiple
            best_idx = best_idxs[0]
            position = int(pred_ranks[best_idx])  # 1-based position
            mrr = 1.0 / position

        per_query_metrics.append(
            {
                "id": item.get("id"),
                "ndcg@1": ndcg_vals["ndcg@1"],
                "ndcg@3": ndcg_vals["ndcg@3"],
                "ndcg@5": ndcg_vals["ndcg@5"],
                "spearman": sp,
                "kendall": kd,
                "mrr": mrr,
                "n_candidates": len(cands),
            }
        )

    return pd.DataFrame(per_query_metrics)


# ---------------------------
# Run both models and compare
# ---------------------------
def main():
    # auto-fallback if GPU not available, or user sets DEVICE="cpu"
    import torch

    device_to_use = DEVICE if (DEVICE == "cpu" or torch.cuda.is_available()) else "cpu"
    print("Using device:", device_to_use)

    model_base = load_model(MODEL_BASE, device_to_use)
    model_fine = load_model(MODEL_FINE, device_to_use)

    print("Evaluating base model...")
    df_base = evaluate_model_on_dataset(model_base, dataset, batch_size=BATCH_SIZE)
    print("Evaluating fine-tuned model...")
    df_fine = evaluate_model_on_dataset(model_fine, dataset, batch_size=BATCH_SIZE)

    # Merge dataframes for comparison
    df_base = df_base.set_index("id")
    df_fine = df_fine.set_index("id")
    combined = df_base.add_suffix("_base").join(
        df_fine.add_suffix("_fine"), how="outer"
    )

    # Compute per-metric averages across queries (ignore NaNs)
    metrics = ["ndcg@1", "ndcg@3", "ndcg@5", "spearman", "kendall", "mrr"]
    summary = {}
    for m in metrics:
        summary[f"{m}_base_mean"] = combined[f"{m}_base"].mean(skipna=True)
        summary[f"{m}_fine_mean"] = combined[f"{m}_fine"].mean(skipna=True)
        summary[f"{m}_delta"] = summary[f"{m}_fine_mean"] - summary[f"{m}_base_mean"]

    summary_df = pd.DataFrame([summary]).T
    summary_df.columns = ["value"]
    print("\nPer-model averages and deltas:")
    print(summary_df)

    # Paired significance test on NDCG@5 where both not NaN
    mask = combined["ndcg@5_base"].notna() & combined["ndcg@5_fine"].notna()
    if mask.sum() > 0:
        stat, p = wilcoxon(
            combined.loc[mask, "ndcg@5_base"], combined.loc[mask, "ndcg@5_fine"]
        )
        print(
            f"\nWilcoxon paired test on NDCG@5: stat={stat:.4f}, p={p:.4e} (H0: medians equal)"
        )
    else:
        print("\nNot enough pairs for significance test on NDCG@5.")

    # Save detailed CSV
    out = combined.reset_index()
    out.to_csv(SAVE_CSV, index=False, encoding="utf-8")
    print(f"\nSaved per-query comparison to {SAVE_CSV}")


if __name__ == "__main__":
    main()
