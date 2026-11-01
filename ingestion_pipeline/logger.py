import logging
from pathlib import Path
from datetime import datetime

# Create logs directory if it doesn't exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Generate timestamp for log filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

LOG_FILE = LOG_DIR / f"ingestion_{timestamp}.log"


def get_logger(name="avrum_ingestion"):

    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # File handler
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")

    # Console handler
    console_handler = logging.StreamHandler()

    # Log formatting
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# import logging
# import os

# from datetime import datetime


# def setup_logger():

#     # Create a unique ID for this ingestion run
#     run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

#     # Create logs directory if it doesn't exist
#     os.makedirs("logs", exist_ok=True)

#     # Generate timestamped log filename
#     log_filename = f"logs/ingestion_{run_id}.log"

#     # Configure log format
#     log_format = (
#         "%(asctime)s | %(levelname)s | "
#         "%(name)s | %(message)s"
#     )

#     formatter = logging.Formatter(log_format)

#     # File logging
#     file_handler = logging.FileHandler(
#         log_filename,
#         encoding="utf-8"
#     )

#     file_handler.setFormatter(formatter)

#     # Terminal logging
#     console_handler = logging.StreamHandler()

#     console_handler.setFormatter(formatter)

#     # Configure root logger
#     root_logger = logging.getLogger()

#     root_logger.setLevel(logging.INFO)

#     root_logger.handlers.clear()

#     root_logger.addHandler(file_handler)
#     root_logger.addHandler(console_handler)

#     logging.info("Logging system initialized")
#     logging.info("Log file: %s", log_filename)

#     return run_id
