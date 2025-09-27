# fine_tune_small.py
import csv
import os

import torch

# Make allocator more flexible (do this before importing heavy libs sometimes)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

# If you want to force CPU for debugging:
# device = "cpu"
# Otherwise let SentenceTransformers pick GPU automatically
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# small model (you already used this)
model_name = "intfloat/multilingual-e5-large"
# word_embedding_model = models.Transformer(model_name, max_seq_length=512)
# pooling_method = word_embedding_model.get_word_embedding_dimension()
# pooling_model = models.Pooling(pooling_method)
model = SentenceTransformer(model_name, device=device)

# free any cached memory before creating dataloaders / optimizer
if device == "cuda":
    torch.cuda.empty_cache()

# Load data
with open(
    "dataset_collection/data/raw_es_results.csv", "r", newline="", encoding="utf-8"
) as f:
    reader = csv.reader(f)
    rows = list(reader)
    # skip header if present
    train_sentences = [row[1] for row in rows[1:]] if len(rows) > 0 else []

# Create InputExamples
train_data = [InputExample(texts=[s, s]) for s in train_sentences]

# **IMPORTANT**: drastically reduce batch_size for small GPU
# try 16, if still OOM -> 8 -> 4 -> 2
BATCH_SIZE = 1

train_dataloader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,  # set 0 if you hit dataloader issues
    pin_memory=(device == "cuda"),
)

train_loss = losses.MultipleNegativesRankingLoss(model)

# Fit: enable AMP (mixed precision). This reduces memory usage a lot.
# Also keep show_progress_bar True for feedback.
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    show_progress_bar=True,
    # weight_decay=0.02,
    use_amp=True,
)

model.save("fine_tune/output/multilingual-e5-small-finetuned-using-simcse")
