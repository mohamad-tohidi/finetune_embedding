# FineTune Embedding Model

## 📌 Overview

**Embedding FineTune** is a repository designed to streamline the process of fine-tuning **E5 embedding model** using data extracted from **Elasticsearch**.

The repository is organized into two main folders:

1. **Dataset Collection (`dataset_collection/`)**  
   This folder contains script to extract data from Elasticsearch index and save it into a structured CSV file. The CSV dataset will later be used for training the embedding model.

2. **Fine-Tune (`fine_tune/`)**  
   This folder contains fine-tuning the **E5 embedding model** using the dataset collected from Elasticsearch. It handles training, evaluation, and saving the fine-tuned embeddings for downstream tasks.
