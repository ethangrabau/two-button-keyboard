"""
Pattern matcher for Two-Button Keyboard.
Maps L/R patterns to words based on frequency.
"""

import json
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Only show warnings and errors

class PatternMatcher:
    def __init__(self, frequency_file: str):
        self.patterns: Dict[str, List[str]] = {}
        self.word_frequencies: Dict[str, float] = {}
        self.load_frequencies(frequency_file)
        
    def load_frequencies(self, filename: str):
        """Load word frequencies and build pattern index."""
        try:
            with open(filename, 'r') as f:
                self.word_frequencies = json.load(f)
                
            # Build pattern index
            for word, freq in self.word_frequencies.items():
                pattern = self.get_pattern(word)
                if pattern not in self.patterns:
                    self.patterns[pattern] = []
                self.patterns[pattern].append(word)
                
            logger.info(f"Loaded {len(self.word_frequencies)} words")
            logger.info(f"Generated {len(self.patterns)} unique patterns")
            
        except Exception as e:
            logger.error(f"Error loading frequencies: {e}")
            raise
            
    def get_pattern(self, word: str) -> str:
        """Convert a word to its L/R pattern."""
        left_keys = set('qwertasdfgzxcv')
        return ''.join('L' if c.lower() in left_keys else 'R' 
                      for c in word)
    
    def predict(self, pattern: str, max_results: int = 5) -> List[str]:
        """Get word predictions for a pattern."""
        if not pattern:
            return []
            
        # Find exact matches
        matches = self.patterns.get(pattern, [])
        
        # Sort by frequency
        predictions = sorted(
            matches,
            key=lambda w: self.word_frequencies.get(w, 0),
            reverse=True
        )[:max_results]
        
        logger.info(f"Pattern '{pattern}' -> {predictions}")
        return predictions
    
    def get_word_frequency(self, word: str) -> float:
        """Get the frequency score for a word."""
        return self.word_frequencies.get(word, 0)
        
    def get_pattern_stats(self) -> Dict:
        """Get statistics about loaded patterns."""
        return {
            'total_patterns': len(self.patterns),
            'total_words': len(self.word_frequencies)
        }