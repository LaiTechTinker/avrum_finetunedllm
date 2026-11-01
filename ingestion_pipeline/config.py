import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")

DATABASE_NAME = os.getenv("DATABASE_NAME", "Avron")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "messages")

if not MONGODB_URI:
    raise ValueError("MONGODB_URI is missing from environment variables")