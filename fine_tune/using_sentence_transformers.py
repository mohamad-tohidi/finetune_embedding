import csv
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


data_files="dataset_collection/data/raw_es_results.csv"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


model_name = "Alibaba-NLP/gte-multilingual-base"
model = SentenceTransformer(model_name, device=device, trust_remote_code=True)


train_dataset = load_dataset("csv", data_files=data_files, cache_dir=None)

print(train_dataset["train"].column_names) 

# with open("dataset_collection/data/evaluation_data.json", "r", encoding="utf-8") as ef:
#     eval_items = json.load(ef)
# eval_qs = []
# eval_ps = []
# eval_scores = []
# for item in eval_items:
#     q = item.get("question", "").strip()
#     rels = item.get("relevant_questions", [])
#     for rel in rels:
#         p = rel.get("content", "").strip()
#         rank = rel.get("rank", 1)
#         score = max(0.0, 5.0 - 0.5 * (rank - 1))
#         if q and p:
#             eval_qs.append(q)
#             eval_ps.append(p)
#             eval_scores.append(score)
# evaluator = EmbeddingSimilarityEvaluator(
#     eval_qs, eval_ps, eval_scores, name="val_q_rel_pairs"
# )


# train_loss = losses.MultipleNegativesRankingLoss(model)

# args = SentenceTransformerTrainingArguments(
#     gradient_accumulation_steps=16,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=3,
#     save_strategy="steps",
#     save_steps=500,
#     eval_strategy="steps",
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_loss",
#     greater_is_better=False,
#     save_total_limit=1,
#     output_dir="fine_tune/output/gte-multilingual-base-finetuned-using-simcse",
# )

# trainer = SentenceTransformerTrainer(
#     model=model,
#     args=args,
#     train_dataset=train_dataset,
#     evaluator=evaluator,
#     loss=train_loss,
# )

# trainer.train()
