import csv
import logging

from elasticsearch import Elasticsearch, helpers

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

    try:
        # Use helpers.scan to get all docs
        results = []
        for doc in helpers.scan(
            es_client,
            index=INDEX_NAME,
            query={"query": {"match_all": {}}},
            preserve_order=False,
        ):
            _id = doc["_id"]
            source = doc["_source"]

            # If you want to extract specific fields, adjust here
            results.append({"_id": _id, **source})

        logger.info(f"Collected {len(results)} documents from index: {INDEX_NAME}")

        # Save to CSV
        csv_file = "dataset_collection/data/all_es_data.csv"

        # Dynamically get all possible fieldnames from sources
        fieldnames = set()
        for r in results:
            fieldnames.update(r.keys())
        fieldnames = list(fieldnames)

        with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        logger.info(f"Saved {len(results)} documents to {csv_file}")

    except Exception as e:
        logger.error(f"Error occurred: {e}")


if __name__ == "__main__":
    main()
