"""
Type definitions for phrase matching system.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class WordCandidate:
    word: str
    frequency_score: float
    pattern_confidence: float
    context_score: float
    cached: bool = False

@dataclass
class PositionCandidates:
    position: int
    pattern: str
    candidates: List[WordCandidate]

class PhraseCache:
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, List[WordCandidate]] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        
    def get(self, key: str) -> Optional[List[WordCandidate]]:
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
        
    def set(self, key: str, value: List[WordCandidate]):
        if len(self.cache) >= self.max_size:
            remove_count = self.max_size // 5
            for _ in range(remove_count):
                self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value
        
    def get_stats(self) -> Dict:
        return {
            'size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }