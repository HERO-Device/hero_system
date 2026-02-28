"""
Consultation configuration settings.
"""


class ConsultConfig:
    """Configuration for consultation session."""

    def __init__(self, speech=True):
        self.speech = speech
        self.text = True
        self.output_lang = 'en'
        self.input_lang = 'en'


# MongoDB connection - LAZY LOAD (only connect when needed)
_client = None


def get_mongo_client():
    """
    Get MongoDB client (lazy initialization).
    Only connects when actually needed.
    """
    global _client

    if _client is not None:
        return _client

    try:
        from pymongo import MongoClient

        # TODO: Load credentials from environment variables
        username = 'rosemaryellery'
        password = '27YOjZirWfNwcCc1'

        _client = MongoClient(
            f'mongodb+srv://{username}:{password}@cluster0.l7w57ga.mongodb.net/'
            f'?retryWrites=true&w=majority&appName=Cluster0',
            serverSelectionTimeoutMS=5000  # 5 second timeout
        )

        # Test connection
        _client.admin.command('ping')
        print("✓ MongoDB connected")

        return _client

    except Exception as e:
        print(f"⚠ Warning: MongoDB connection failed: {e}")
        print("  → Running in local-only mode")
        return None


# For backwards compatibility (don't connect on import!)
client = None
