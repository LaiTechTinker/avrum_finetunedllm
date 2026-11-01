
import logging
from itertools import groupby

logger = logging.getLogger(__name__)


def process_conversations(messages):

    logger.info("Starting conversation processing...")

    processed_count = 0
    skipped_count = 0

    # Messages must already be sorted by conversation_id
    for conversation_id, group in groupby(
        messages,
        key=lambda message: message["conversation_id"]
    ):

        # Only load one conversation into memory at a time
        conversation = list(group)

        # Keep only valid user and assistant messages
        conversation = [
            {
                "role": message["role"],
                "content": message["content"]
            }
            for message in conversation
            if message.get("role") in ["user", "assistant"]
            and isinstance(message.get("content"), str)
            and message["content"].strip()
        ]

        # Remove assistant messages appearing before the first user message
        while conversation and conversation[0]["role"] == "assistant":
            conversation.pop(0)

        # Remove unanswered user messages at the end
        while conversation and conversation[-1]["role"] == "user":
            conversation.pop()

        # Require at least one user message followed by an assistant response
        has_valid_pair = any(
            conversation[i]["role"] == "user"
            and conversation[i + 1]["role"] == "assistant"
            for i in range(len(conversation) - 1)
        )

        if not has_valid_pair:
            skipped_count += 1
            continue

        training_example = {
            "messages": conversation
        }

        processed_count += 1

        if processed_count % 100 == 0:
            logger.info(
                "Processed %s conversations",
                processed_count
            )

        yield training_example

    logger.info(
        "Conversation processing completed. "
        "Processed: %s | Skipped: %s",
        processed_count,
        skipped_count
    )
