"""
Cache manager for Two-Button Keyboard prediction system.
Handles caching of pattern matches and phrase predictions.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time

@dataclass
class CacheEntry:
    value: Any
    timestamp: float
    access_count: int

class PredictionCache:
    def __init__(self, max_size: int = 1000, ttl: float = 3600):
        self.pattern_cache: Dict[str, CacheEntry] = {}
        self.phrase_cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
        
    def get_pattern(self, pattern: str) -> Optional[List[str]]:
        """Get cached pattern prediction if available."""
        entry = self.pattern_cache.get(pattern)
        if not entry:
            return None
            
        # Check TTL
        if time.time() - entry.timestamp > self.ttl:
            del self.pattern_cache[pattern]
            return None
            
        entry.access_count += 1
        return entry.value
        
    def set_pattern(self, pattern: str, predictions: List[str]):
        """Cache pattern predictions with LRU eviction."""
        self._ensure_capacity(self.pattern_cache)
        self.pattern_cache[pattern] = CacheEntry(
            value=predictions,
            timestamp=time.time(),
            access_count=1
        )
        
    def get_phrase(self, key: str) -> Optional[List[str]]:
        """Get cached phrase prediction if available."""
        entry = self.phrase_cache.get(key)
        if not entry:
            return None
            
        if time.time() - entry.timestamp > self.ttl:
            del self.phrase_cache[key]
            return None
            
        entry.access_count += 1
        return entry.value
        
    def set_phrase(self, key: str, predictions: List[str]):
        """Cache phrase predictions with LRU eviction."""
        self._ensure_capacity(self.phrase_cache)
        self.phrase_cache[key] = CacheEntry(
            value=predictions,
            timestamp=time.time(),
            access_count=1
        )
        
    def _ensure_capacity(self, cache: Dict):
        """Ensure cache doesn't exceed max size using LRU policy."""
        if len(cache) >= self.max_size:
            # Remove 20% least recently used entries
            entries = sorted(
                cache.items(),
                key=lambda x: (x[1].access_count, -x[1].timestamp)
            )
            for key, _ in entries[:max(1, len(cache) // 5)]:
                del cache[key]
                
    def clear(self):
        """Clear all caches."""
        self.pattern_cache.clear()
        self.phrase_cache.clear()
