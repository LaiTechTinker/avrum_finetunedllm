
import logging

from pymongo import MongoClient

from ingestion_pipeline.config import (
    MONGODB_URI,
    DATABASE_NAME,
    COLLECTION_NAME
)


logger = logging.getLogger(__name__)

client = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=5000
)

database = client[DATABASE_NAME]

messages_collection = database[COLLECTION_NAME]


def test_connection():

    try:

        client.admin.command("ping")

        logger.info("MongoDB connection successful")
        logger.info("Database: %s", DATABASE_NAME)
        logger.info("Collection: %s", COLLECTION_NAME)

        return True

    except Exception:

        logger.exception("MongoDB connection failed")

        return False
