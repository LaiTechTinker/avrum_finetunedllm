
import logging
import os

from datetime import datetime


def setup_logger():

    # Create a unique ID for this ingestion run
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Generate timestamped log filename
    log_filename = f"logs/training_{run_id}.log"

    # Configure log format
    log_format = (
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    )

    formatter = logging.Formatter(log_format)

    # File logging
    file_handler = logging.FileHandler(
        log_filename,
        encoding="utf-8"
    )

    file_handler.setFormatter(formatter)

    # Terminal logging
    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()

    root_logger.setLevel(logging.INFO)

    root_logger.handlers.clear()

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info("Logging system initialized")
    logging.info("Log file: %s", log_filename)

    return run_id
