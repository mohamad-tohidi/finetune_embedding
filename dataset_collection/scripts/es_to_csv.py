import csv
import logging

from elasticsearch import Elasticsearch

from dataset_collection.utils import ES_PASSWORD, ES_URL, ES_USER, INDEX_NAME

# Configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("es_to_csv")


def main():
    # Connect to ES
    es_client = Elasticsearch(
        hosts=[{"host": ES_URL, "port": 9200, "scheme": "https"}],
        basic_auth=(ES_USER, ES_PASSWORD),
        verify_certs=False,
    )
    logger.info(f"Connected to ES: {ES_URL}")

    # Build the query
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

    # Run the search
    try:
        response = es_client.search(index=INDEX_NAME, body=query)
        logger.info(f"Searched through index: {INDEX_NAME}")

        # Collect all results
        results = []
        for bucket in response["aggregations"]["by_category"]["buckets"]:
            category = bucket["key"]
            if category == "هیچکدام":
                continue
            docs = bucket["top_docs"]["hits"]["hits"]
            for doc in docs:
                _id = doc["_id"]
                content = doc["_source"]["question"]["text"]["fa"]
                results.append({"_id": _id, "content": content, "category": category})
        logger.info(f"Collected all results from index: {INDEX_NAME}")

        # Save to CSV
        csv_file = "dataset_collection/data/raw_es_results.csv"
        with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["_id", "content", "category"])
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Saved {len(results)} documents to {csv_file}")

    except Exception as e:
        logger.error(f"Error occured: {e}")


if __name__ == "__main__":
    main()
