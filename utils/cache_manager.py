# soufianeelseflo-aiagency/utils/cache_manager.py
import os
import time
import logging
from typing import Optional, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CacheManager:
    """Manages a smart cache for high-value data to boost agency profits."""
    def __init__(self):
        self.cache: Dict[str, Dict] = {}
        self.cache_ttl = int(os.getenv("CACHE_TTL", "3600"))  # Default 1 hour
        self.max_size = int(os.getenv("CACHE_MAX_SIZE", "1000"))  # Max items
        self.hits = {}  # Track usage for profit priority
        logging.info(f"CacheManager initialized with TTL: {self.cache_ttl}s, max size: {self.max_size}")

    def get(self, key: str) -> Optional[dict]:
        """Retrieve cached item if fresh and valuable."""
        if key in self.cache:
            cached_data = self.cache[key]
            if time.time() - cached_data["timestamp"] < self.cache_ttl:
                self.hits[key] = self.hits.get(key, 0) + 1
                logging.info(f"Cache hit for key: {key}, hits: {self.hits[key]}")
                return cached_data["value"]
            else:
                del self.cache[key]
                del self.hits[key]
                logging.info(f"Cache expired for key: {key}")
        return None

    def set(self, key: str, value: dict) -> None:
        """Store item, prioritize profit-driven data."""
        if len(self.cache) >= self.max_size:
            # Evict least-used item
            least_used = min(self.hits.items(), key=lambda x: x[1])[0]
            del self.cache[least_used]
            del self.hits[least_used]
            logging.info(f"Evicted least-used cache item: {least_used}")
        self.cache[key] = {"value": value, "timestamp": time.time()}
        self.hits[key] = self.hits.get(key, 0) + 1
        logging.info(f"Cached result for key: {key}")

    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.hits.clear()
        logging.info("Cache cleared.")

    def size(self) -> int:
        """Return cache size."""
        return len(self.cache)

    def get_profit_insights(self) -> str:
        """Report cache hits for profit analysis."""
        top_hits = sorted(self.hits.items(), key=lambda x: x[1], reverse=True)[:5]
        return f"Top cached items: {', '.join([f'{k}: {v} hits' for k, v in top_hits])}"