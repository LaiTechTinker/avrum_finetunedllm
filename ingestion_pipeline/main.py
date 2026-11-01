
import logging

from ingestion_pipeline.logger import setup_logger
from ingestion_pipeline.database import test_connection
from ingestion_pipeline.extractor import extract_messages
from ingestion_pipeline.processor import process_conversations
from ingestion_pipeline.exporter import export_conversations


logger = logging.getLogger(__name__)


def main():

    run_id = setup_logger()

    logger.info("=" * 50)
    logger.info("AVRUM DATA INGESTION PIPELINE STARTED")
    logger.info("Run ID: %s", run_id)
    logger.info("=" * 50)

    try:

        logger.info("Checking MongoDB connection")

        if not test_connection():
            logger.error("Database connection failed")
            return

        logger.info("MongoDB connection successful")

        logger.info("Starting message extraction")

        messages = extract_messages()

        logger.info("Starting conversation processing")

        conversations = process_conversations(messages)

        logger.info("Starting JSON export")

        exported_count = export_conversations(
            conversations,
            run_id
        )

        logger.info("=" * 50)
        logger.info("INGESTION PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("Total conversations exported: %s", exported_count)
        logger.info("=" * 50)

    except Exception:

        logger.exception("Ingestion pipeline failed")

        raise


if __name__ == "__main__":
    main()
