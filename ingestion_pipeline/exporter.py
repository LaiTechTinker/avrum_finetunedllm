
import json
import logging
import os

logger = logging.getLogger(__name__)

OUTPUT_DIR = "output"
MAX_FILE_SIZE = 5 * 1024 * 1024


def export_conversations(conversations, run_id):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    file_number = 1
    exported_count = 0
    current_file = None
    current_size = 2  # Opening bracket and newline
    records_in_file = 0

    def close_current_file():

        nonlocal current_file

        if current_file:
            current_file.write("\n]")
            current_file.close()
            current_file = None

    def open_new_file():

        nonlocal file_number, current_file
        nonlocal current_size, records_in_file

        filename = os.path.join(
            OUTPUT_DIR,
            f"conversations_{run_id}_{file_number:03d}.json"
        )

        current_file = open(
            filename,
            "w",
            encoding="utf-8"
        )

        current_file.write("[\n")

        current_size = 2
        records_in_file = 0

        logger.info("Created output file: %s", filename)

        file_number += 1

    try:

        for conversation in conversations:

            json_record = json.dumps(
                conversation,
                ensure_ascii=False,
                separators=(",", ":")
            )

            record_size = len(json_record.encode("utf-8"))

            # Account for the final closing bracket and newline
            separator_size = 2 if records_in_file > 0 else 0

            required_size = (
                current_size
                + separator_size
                + record_size
                + 2
            )

            # Rotate file if the next record won't fit
            if required_size > MAX_FILE_SIZE:

                if records_in_file > 0:
                    close_current_file()
                    current_file = None

                # Check whether one record can fit in an empty file
                if record_size + 4 > MAX_FILE_SIZE:

                    logger.warning(
                        "Skipping oversized conversation: %s bytes",
                        record_size
                    )
                    continue

                open_new_file()

                separator_size = 0

            elif current_file is None:

                open_new_file()

            if records_in_file > 0:
                current_file.write(",\n")

            current_file.write(json_record)

            current_size += separator_size + record_size

            records_in_file += 1
            exported_count += 1

            if exported_count % 100 == 0:
                logger.info(
                    "Exported %s conversations",
                    exported_count
                )

        close_current_file()

        logger.info(
            "Export finished | Total conversations exported: %s",
            exported_count
        )

        return exported_count

    except Exception:

        logger.exception("An error occurred during JSON export")
        raise

    finally:
        close_current_file()
