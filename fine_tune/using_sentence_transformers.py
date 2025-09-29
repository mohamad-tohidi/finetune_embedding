import json

import torch
from datasets import load_dataset
from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator



data_files="dataset_collection/data/all_es_data.csv"
output_dir="fine_tune/output/gte-multilingual-base-finetuned-using-simcse"
evaluation_data = "dataset_collection/data/evaluation_data.json"


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


model_name = "Alibaba-NLP/gte-multilingual-base"
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


train_dataset_raw = load_dataset("csv", data_files=data_files, cache_dir=None)
train_dataset = pre_process_dataset(train_dataset_raw, text_col="question.text.fa")


train_dataset = train_dataset.map(
    lambda example: {"anchor": example["text"], "positive": example["text"]},
    remove_columns=["text"]
)

print(train_dataset)



with open(evaluation_data, "r", encoding="utf-8") as ef:
    eval_items = json.load(ef)
eval_qs = []
eval_ps = []
eval_scores = []
for item in eval_items:
    q = item.get("question", "").strip()
    rels = item.get("relevant_questions", [])
    for rel in rels:
        p = rel.get("content", "").strip()
        rank = rel.get("rank", 1)
        score = max(0.0, 5.0 - 0.5 * (rank - 1))
        if q and p:
            eval_qs.append(q)
            eval_ps.append(p)
            eval_scores.append(score)


evaluator = EmbeddingSimilarityEvaluator(
    eval_qs, eval_ps, eval_scores, name="val_q_rel_pairs", main_similarity="cosine"
)

train_loss = losses.MultipleNegativesRankingLoss(model)

args = SentenceTransformerTrainingArguments(
    gradient_accumulation_steps=16,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_strategy="steps",
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_val_q_rel_pairs_spearman_cosine",
    greater_is_better=True,
    save_total_limit=1,
    output_dir=output_dir,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    evaluator=evaluator,
    loss=train_loss,
)

trainer.train()