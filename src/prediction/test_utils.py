"""
Test utilities and shared data for Two-Button Keyboard tests.
"""

from typing import List, Dict, Any
import statistics
from pathlib import Path

# Expected pattern-to-word mappings for verification
EXPECTED_MATCHES = {
    # Single letter patterns
    "R": ["hi", "me", "no"],
    "L": ["the", "we"],
    
    # Two letter patterns
    "RR": ["hi", "me", "no"],
    "RL": ["is", "it"],
    "LR": ["the", "we"],
    "LL": ["yes"],
    
    # Common word patterns
    "RRL": ["how", "has"],
    "LRL": ["the", "that"],
    "RLL": ["you", "your"],
    "LRR": ["this", "there"],
    
    # Question starts
    "LRLL": ["what", "when", "where"],
    "RRL LLL": ["how are"],
    "LRLL RL": ["what is", "where is"],
    
    # Common phrases
    "RR RRL": ["hi how", "hi now"],
    "RR RRL LLL": ["hi how are", "hi now its"],
    "RR RRL LLL RRR": ["hi how are you"]
}

# Test conversations with expected responses
TEST_CONVERSATIONS = [
    {
        "input": "Hello!",
        "pattern": "RR RRL LLL RRR",
        "expected": "hi how are you"
    },
    {
        "input": "Hi how are you?",
        "pattern": "RR LRLL",
        "expected": "im fine"
    },
    {
        "input": "What's up?",
        "pattern": "RRL LRRR",
        "expected": "not much"
    }
]

class TestResults:
    def __init__(self):
        self.latencies: List[float] = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.accuracy_scores: List[float] = []
        self.pattern_matches: Dict[str, bool] = {}
        
    def add_latency(self, ms: float):
        self.latencies.append(ms)
        
    def add_accuracy(self, score: float):
        self.accuracy_scores.append(score)
        
    def add_pattern_match(self, pattern: str, success: bool):
        self.pattern_matches[pattern] = success
        
    def get_stats(self) -> Dict[str, Any]:
        if not self.latencies:
            return {}
            
        return {
            "latency": {
                "min": min(self.latencies),
                "max": max(self.latencies),
                "mean": statistics.mean(self.latencies),
                "median": statistics.median(self.latencies)
            },
            "accuracy": {
                "mean": statistics.mean(self.accuracy_scores) if self.accuracy_scores else 0,
                "pattern_matches": sum(self.pattern_matches.values()),
                "total_patterns": len(self.pattern_matches)
            },
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
            }
        }