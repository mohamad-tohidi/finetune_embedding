import csv
import logging

from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

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
        csv_file = "dataset_collection/data/all_es_data.csv"

        # Open CSV file once, write incrementally
        with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
            fieldnames = ["_id", "question.text.fa"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Iterate with tqdm progress bar
            for doc in tqdm(
                helpers.scan(
                    es_client,
                    index=INDEX_NAME,
                    query={"query": {"match_all": {}}},
                    preserve_order=False,
                ),
                desc="Exporting documents",
            ):
                _id = doc["_id"]
                source = doc["_source"]

                # Only keep "_id" and "question.text.fa" if exists
                question_text = source.get("question", {}).get("text", {}).get("fa")
                if question_text:
                    writer.writerow({"_id": _id, "question.text.fa": question_text})

        logger.info(f"Export completed. Saved data to {csv_file}")

    except Exception as e:
        logger.error(f"Error occurred: {e}")


if __name__ == "__main__":
    main()
