"""
Enhanced caching implementation for Phi-2 interface.
"""

from typing import Optional, Any, Dict
from collections import OrderedDict, defaultdict
import re
from .logging import perf_logger

class SmartCache:
    """Enhanced LRU Cache with fuzzy matching and pattern-based invalidation."""
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.pattern_index = defaultdict(set)  # Track which cache keys use which patterns
        self.metrics = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "invalidations": 0
        }

    def _normalize_key(self, key: str) -> str:
        """Normalize cache key to increase hit rate."""
        # Remove extra whitespace and lowercase
        key = ' '.join(key.lower().split())
        # Sort candidate words to ensure different orderings match
        parts = key.split(':')
        if len(parts) > 1:
            candidates = parts[1].split(',')
            parts[1] = ','.join(sorted(candidates))
        return ':'.join(parts)

    def _index_patterns(self, key: str) -> None:
        """Extract and index patterns from the key."""
        patterns = re.findall(r'[LR]+', key)
        for pattern in patterns:
            self.pattern_index[pattern].add(key)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with metrics tracking."""
        key = self._normalize_key(key)
        if key in self.cache:
            self.cache.move_to_end(key)
            self.metrics["hits"] += 1
            perf_logger.debug(f"Cache hit for key: {key}")
            return self.cache[key]
        self.metrics["misses"] += 1
        perf_logger.debug(f"Cache miss for key: {key}")
        return None

    def put(self, key: str, value: Any) -> None:
        """Put value in cache with metrics tracking."""
        key = self._normalize_key(key)
        if key in self.cache:
            self.cache.move_to_end(key)
            perf_logger.debug(f"Updated existing key: {key}")
        else:
            perf_logger.debug(f"Adding new key: {key}")
            
        self.cache[key] = value
        self._index_patterns(key)
        
        if len(self.cache) > self.capacity:
            # Remove oldest item and clean up pattern index
            old_key, _ = self.cache.popitem(last=False)
            self.metrics["evictions"] += 1
            perf_logger.debug(f"Evicted key due to capacity: {old_key}")
            for pattern_keys in self.pattern_index.values():
                pattern_keys.discard(old_key)

    def invalidate_pattern(self, pattern: str) -> None:
        """Invalidate all cache entries containing a pattern."""
        keys_to_remove = self.pattern_index.get(pattern, set())
        for key in keys_to_remove:
            if key in self.cache:
                del self.cache[key]
                self.metrics["invalidations"] += 1
        del self.pattern_index[pattern]

    def get_metrics(self) -> Dict:
        """Get cache performance metrics."""
        hit_rate = (self.metrics["hits"] / (self.metrics["hits"] + self.metrics["misses"])) * 100 \
                  if (self.metrics["hits"] + self.metrics["misses"]) > 0 else 0
                  
        return {
            "size": len(self.cache),
            "capacity": self.capacity,
            "hit_rate": f"{hit_rate:.1f}%",
            "hits": self.metrics["hits"],
            "misses": self.metrics["misses"],
            "evictions": self.metrics["evictions"],
            "invalidations": self.metrics["invalidations"],
            "pattern_index_size": sum(len(patterns) for patterns in self.pattern_index.values())
        }