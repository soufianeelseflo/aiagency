import os
import json
import time
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)

class CacheManager:
    def __init__(self):
        # Initialize cache storage (in-memory for simplicity; can be extended to Redis or disk-based)
        self.cache = {}
        self.cache_ttl = int(os.getenv("CACHE_TTL", 3600))  # Default TTL: 1 hour

    def get(self, key: str) -> Optional[dict]:
        """
        Retrieve a cached result if it exists and has not expired.
        """
        if key in self.cache:
            cached_data = self.cache[key]
            if time.time() - cached_data["timestamp"] < self.cache_ttl:
                logging.info(f"Cache hit for key: {key}")
                return cached_data["value"]
            else:
                logging.info(f"Cache expired for key: {key}")
                del self.cache[key]  # Remove expired entry
        return None

    def set(self, key: str, value: dict) -> None:
        """
        Store a result in the cache with a timestamp.
        """
        self.cache[key] = {
            "value": value,
            "timestamp": time.time()
        }
        logging.info(f"Cached result for key: {key}")

    def clear(self) -> None:
        """
        Clear all cached entries.
        """
        self.cache.clear()
        logging.info("Cache cleared.")

    def size(self) -> int:
        """
        Return the number of cached entries.
        """
        return len(self.cache)