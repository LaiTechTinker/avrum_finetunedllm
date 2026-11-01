
import logging
from datetime import datetime
from itertools import islice

from ingestion_pipeline.database import test_connection
from ingestion_pipeline.extractor import extract_messages
from ingestion_pipeline.processor import process_conversations
from ingestion_pipeline.exporter import export_conversations


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():

    print("Starting Avrum Data Ingestion Pipeline...")

    if not test_connection():
        print("Database connection failed.")
        return

    print("\nStarting data extraction...")

    messages = extract_messages()

    print("\nStarting conversation processing...")

    conversations = process_conversations(messages)

    print("\nExporting conversations to JSON...")

    exported_count = export_conversations(conversations)

    print("\nIngestion pipeline completed.")

    print(f"Total conversations exported: {exported_count}")


if __name__ == "__main__":
    main()
