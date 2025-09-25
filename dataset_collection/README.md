# Dataset Collection

**Dataset Collection** is a folder for extracting documents from an Elasticsearch index and saving them to a CSV file.  

It enables fast exporting and structuring of **ES data** for analysis, processing, or downstream tasks.

## How to Customize?
Modify the `query` variable in `scripts/data_loader.py` to change which data is retrieved, how it is aggregated, and which fields are included in the output.

```python
# Elasticsearch aggregation query:
# - Groups documents by 'question.metadata.category.keyword' (top 8 categories)
# - For each category, retrieves top 100 documents
# - Only includes 'question.text.fa' field in results
query = {
    "size": 0,
    "aggs": {
        "by_category": {
            "terms": {"field": "question.metadata.category.keyword", "size": 8},
            "aggs": {
                "top_docs": {
                    "top_hits": {
                        "_source": {"includes": ["question.text.fa"]},
                        "size": 100,
                    }
                }
            },
        }
    },
}
```

Note: **"size"** field in the **"aggs"** is to specify how many ducs to pick from each category.