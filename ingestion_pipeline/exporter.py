
import json
import logging
import os

logger = logging.getLogger(__name__)

OUTPUT_DIR = "output"

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


def export_conversations(conversations):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    file_number = 1
    exported_count = 0

    current_file = None
    current_size = 2  # Opening and closing square brackets: []

    def open_new_file():

        nonlocal file_number, current_size, current_file

        if current_file:
            current_file.write("\n]")
            current_file.close()

        filename = os.path.join(
            OUTPUT_DIR,
            f"conversations_{file_number:03d}.json"
        )

        current_file = open(filename, "w", encoding="utf-8")

        current_file.write("[\n")

        current_size = 2

        logger.info("Created output file: %s", filename)

        file_number += 1

    try:

        for conversation in conversations:

            # Convert conversation into JSON text
            json_record = json.dumps(
                conversation,
                ensure_ascii=False,
                separators=(",", ":")
            )

            # Calculate actual UTF-8 byte size
            record_size = len(json_record.encode("utf-8"))

            # Account for comma and newline between records
            separator_size = 2 if current_size > 2 else 0

            required_size = record_size + separator_size

            # Start a new file if the next record exceeds 5 MB
            if current_file is None or (
                current_size + required_size + 1 > MAX_FILE_SIZE
            ):
                open_new_file()

                separator_size = 0
                required_size = record_size

            # A single record must fit inside the size limit
            if current_size + required_size + 1 > MAX_FILE_SIZE:
                logger.warning(
                    "Skipping conversation because it exceeds 5 MB"
                )
                continue

            if current_size > 2:
                current_file.write(",\n")

            current_file.write(json_record)

            current_size += required_size

            exported_count += 1

            if exported_count % 100 == 0:
                logger.info(
                    "Exported %s conversations",
                    exported_count
                )

        logger.info(
            "Export completed. Total conversations exported: %s",
            exported_count
        )

    finally:

        if current_file:
            current_file.write("\n]")
            current_file.close()

    return exported_count
