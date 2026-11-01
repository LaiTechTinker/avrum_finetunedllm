
import logging

from ingestion.database import messages_collection


logger = logging.getLogger(__name__)


def extract_messages(batch_size=500):

    logger.info("Starting message extraction...")

    projection = {
        "_id": 0,
        "conversation_id": 1,
        "role": 1,
        "content": 1,
        "created_at": 1
    }

    cursor = messages_collection.find(
        {},
        projection
    ).sort([
        ("conversation_id", 1),
        ("created_at", 1)
    ]).batch_size(batch_size)

    extracted_count = 0

    for message in cursor:

        # Skip incomplete documents
        if not message.get("role") or not message.get("content"):
            logger.warning(
                "Skipping message with missing role or content"
            )
            continue

        # Skip unsupported roles
        if message["role"] not in ["user", "assistant"]:
            logger.warning(
                "Skipping unsupported role: %s",
                message["role"]
            )
            continue

        extracted_count += 1

        if extracted_count % 100 == 0:
            logger.info(
                "Extracted %s messages so far",
                extracted_count
            )

        yield {
            "conversation_id": str(message["conversation_id"]),
            "role": message["role"],
            "content": message["content"].strip(),
            "created_at": message["created_at"].isoformat()
        }

    logger.info(
        "Extraction completed. Total messages extracted: %s",
        extracted_count
    )
