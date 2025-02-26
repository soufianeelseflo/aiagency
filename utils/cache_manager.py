import os
import json
import time
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CacheManager:
    def __init__(self):
        """Initialize in-memory cache with configurable TTL (ref: https://docs.python.org/3/library/time.html)."""
        self.cache = {}
        self.cache_ttl = int(os.getenv("CACHE_TTL", 3600))  # Default TTL: 1 hour
        logging.info(f"CacheManager initialized with TTL: {self.cache_ttl} seconds")

    def get(self, key: str) -> Optional[dict]:
        """Retrieve a cached result if it exists and hasn’t expired."""
        if key in self.cache:
            cached_data = self.cache[key]
            if time.time() - cached_data["timestamp"] < self.cache_ttl:
                logging.info(f"Cache hit for key: {key}")
                return cached_data["value"]
            else:
                logging.info(f"Cache expired for key: {key}")
                del self.cache[key]
        return None

    def set(self, key: str, value: dict) -> None:
        """Store a result in the cache with a timestamp."""
        self.cache[key] = {
            "value": value,
            "timestamp": time.time()
        }
        logging.info(f"Cached result for key: {key}")

    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
        logging.info("Cache cleared.")

    def size(self) -> int:
        """Return the number of cached entries."""
        return len(self.cache)