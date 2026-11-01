from pymongo import MongoClient
from ingestion_pipeline.config import (
    MONGODB_URI,
    DATABASE_NAME,
    COLLECTION_NAME
)

client = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=5000
)

database = client[DATABASE_NAME]

messages_collection = database[COLLECTION_NAME]


def test_connection():
    try:
        client.admin.command("ping")

        print("MongoDB connection successful!")

        print(f"Database: {DATABASE_NAME}")
        print(f"Collection: {COLLECTION_NAME}")

        return True

    except Exception as error:
        print(f"MongoDB connection failed: {error}")

        return False