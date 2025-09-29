import json

import torch
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.training_args import BatchSamplers


data_files="dataset_collection/data/all_es_data.csv"
output_dir="fine_tune/output/gte-multilingual-base-finetuned-using-simcse"
evaluation_data = "dataset_collection/data/evaluation_data.json"
model_name = "Alibaba-NLP/gte-multilingual-base"


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


model = SentenceTransformer(model_name, device=device, trust_remote_code=True)


def pre_process_dataset(hf_dataset, text_col="question.text.fa"):
    """
    Convert a HuggingFace Dataset or DatasetDict to a HuggingFace Dataset with a single
    'text' column (cleaned & filtered). Works whether hf_dataset is a Dataset or a
    DatasetDict containing 'train'.
    """
    # Accept either DatasetDict or Dataset
    if isinstance(hf_dataset, dict) and "train" in hf_dataset:
        ds = hf_dataset["train"]
    else:
        ds = hf_dataset

    if text_col not in ds.column_names:
        raise ValueError(f"Column '{text_col}' not found in dataset columns: {ds.column_names}")

    ds = ds.map(
        lambda example: {"text": (str(example.get(text_col, "") or "")).strip()},
        remove_columns=ds.column_names,
        batched=False,
    )

    ds = ds.filter(lambda example: example["text"] != "")

    return ds


print("Loading and preparing training data...")
train_dataset_raw = load_dataset("csv", data_files=data_files, cache_dir=None)
train_dataset = pre_process_dataset(train_dataset_raw, text_col="question.text.fa")


# This prepares the data for a SimCSE-like unsupervised training setup.
train_dataset = train_dataset.map(
    lambda example: {"anchor": example["text"], "positive": example["text"]},
    remove_columns=["text"]
)

# ======================================================================
# ADD THIS PART TO LIMIT THE DATASET SIZE
# ======================================================================
num_training_samples = 20000 

# Use .select() to create a smaller subset of the dataset
if len(train_dataset) > num_training_samples:
    train_dataset = train_dataset.select(range(num_training_samples))
# ======================================================================


print("Training data prepared (and limited):")
print(train_dataset)

print("\nLoading and preparing evaluation data...")
with open(evaluation_data, "r", encoding="utf-8") as f:
    eval_data_json = json.load(f)

eval_sentences1 = []
eval_sentences2 = []
eval_scores = []

for item in eval_data_json:
    question = item.get("question", "").strip()
    if not question:
        continue

    for relevant_item in item.get("relevant_questions", []):
        relevant_question = relevant_item.get("content", "").strip()
        rank = relevant_item.get("rank")

        if relevant_question and rank is not None:
            # Convert rank to a similarity score. A higher rank means lower similarity.
            # Rank 1 -> 5.0, Rank 2 -> 4.5, etc.
            score = max(0.0, 5.0 - 0.5 * (rank - 1))
            eval_sentences1.append(question)
            eval_sentences2.append(relevant_question)
            eval_scores.append(score)

print(f"Created {len(eval_sentences1)} sentence pairs for evaluation.")

# ======================================================================================
# Evaluator Setup
# ======================================================================================
# The EmbeddingSimilarityEvaluator computes the cosine similarity between sentence pairs
# from the model and compares it to a gold standard score (eval_scores) using
# Spearman's rank correlation. This tells us how well the model's rankings match the desired rankings.
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=eval_sentences1,
    sentences2=eval_sentences2,
    scores=eval_scores,
    name="val_q_rel_pairs", 
    main_similarity="cosine"
)

# ======================================================================================
# Loss Function
# ======================================================================================
# MultipleNegativesRankingLoss is a great choice for this unsupervised setup.
# It pulls all "positive" sentences from a batch and treats all others as negatives.
train_loss = losses.MultipleNegativesRankingLoss(model)

# ======================================================================================
# Training Arguments
# ======================================================================================
# **CRITICAL FIX**: The key to making evaluation work is setting the correct
# `metric_for_best_model`. The `EmbeddingSimilarityEvaluator` does not produce a 'loss'.
# It produces a correlation score.
# The name of this metric is constructed as: f"eval_{evaluator.name}_{evaluator.main_similarity}_spearman"
# In our case, this will be: "eval_val_q_rel_pairs_cosine_spearman".
# We also set `greater_is_better=True` because a higher correlation score is better.

metric_name = "eval_val_q_rel_pairs_spearman_cosine"
print(f"\nThe model will be saved based on the best score for the metric: '{metric_name}'")

args = SentenceTransformerTrainingArguments(
    report_to=None,
    # Training-specific arguments
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    # Evaluation-specific arguments
    eval_strategy="steps",
    eval_steps=100,
    per_device_eval_batch_size=2,
    # Save-specific arguments
    save_strategy="steps",
    save_steps=500,
    save_total_limit=1,
    # Arguments for tracking the best model
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    greater_is_better=True, # For Spearman correlation, higher is better
    # General arguments
    output_dir=output_dir,
    # Batch sampler to ensure batches have unique texts, which is beneficial for MultipleNegativesRankingLoss
    # batch_sampler=BatchSamplers.NO_DUPLICATES,
)

# ======================================================================================
# Trainer Setup and Execution
# ======================================================================================
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    evaluator=evaluator,
    loss=train_loss,
)

print("\nStarting training...")
trainer.train()

# ======================================================================================
# Final Model Save
# ======================================================================================
# The trainer saves the best model checkpoint automatically during training.
# You can also save the final trained model state like this.
print("Training finished. Saving the final model.")
model.save(output_dir)
print(f"Model saved to {output_dir}")
