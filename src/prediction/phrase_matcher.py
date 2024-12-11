"""
Phrase Pattern Matcher for Two-Button Keyboard Interface.
Handles multi-word pattern sequences with basic context awareness.
"""

from typing import Dict, List, Tuple, Optional
from pattern_matcher import PatternMatcher

class PhraseMatcher:
    def __init__(self, word_matcher: PatternMatcher):
        self.word_matcher = word_matcher
        # Common phrase patterns and their expected words
        self.common_phrases = {
            "RR RRL LLL RRR": ["hi how are you"],
            "RLRRR RRL LLL RRR": ["hello how are you"],
            "RLR RRL LLL RRR": ["hey how are you"],
            "RRL LLL RRR LRRRL": ["how are you doing"],
            "LRLL RL LRL": ["what is the"]
        }
        
    def split_phrase_pattern(self, pattern: str) -> List[str]:
        """Split a phrase pattern into individual word patterns."""
        return [p.strip() for p in pattern.split() if p.strip()]
    
    def verify_phrase(self, phrase: str, pattern: str) -> bool:
        """Verify that a phrase matches a given pattern sequence."""
        words = phrase.strip().split()
        patterns = self.split_phrase_pattern(pattern)
        
        if len(words) != len(patterns):
            return False
            
        for word, expected_pattern in zip(words, patterns):
            word_pattern = self.word_matcher.get_pattern(word)
            if word_pattern != expected_pattern:
                print(f"Pattern mismatch: '{word}' -> '{word_pattern}' (expected '{expected_pattern}')")
                return False
        
        return True
    
    def get_pattern_for_phrase(self, phrase: str) -> str:
        """Convert a phrase to its L/R pattern sequence."""
        words = phrase.strip().split()
        patterns = [self.word_matcher.get_pattern(word) for word in words]
        return ' '.join(patterns)
    
    def predict_phrase(self, pattern: str, context: List[str] = None) -> List[Tuple[str, float]]:
        """
        Get phrase predictions for a pattern sequence.
        Uses common phrases and context to improve predictions.
        """
        # First check if this is a known phrase pattern
        for known_pattern, phrases in self.common_phrases.items():
            if pattern == known_pattern:
                return [(phrase, 1.0) for phrase in phrases]
        
        patterns = self.split_phrase_pattern(pattern)
        if not patterns:
            return []
            
        # Get predictions for each position with context
        predictions = []
        built_context = []  # Words we've predicted so far
        
        for i, p in enumerate(patterns):
            # Get words that match this pattern
            matches = self.word_matcher.predict(p, max_results=5)
            if not matches:
                return []
                
            # Use position and built context to select best word
            if i == 0:  # First word
                word = self._select_first_word(matches, context)
            elif i == len(patterns) - 1:  # Last word
                word = self._select_last_word(matches, built_context)
            else:  # Middle words
                word = self._select_middle_word(matches, built_context, patterns[i+1:])
                
            predictions.append(word)
            built_context.append(word)
            
        return [(' '.join(predictions), 1.0)]
    
    def _select_first_word(self, words: List[str], context: List[str]) -> str:
        """Select best first word based on context."""
        # Prefer greeting words at start of conversation
        greeting_words = {'hi', 'hello', 'hey'}
        if not context:  # No previous messages
            for word in words:
                if word.lower() in greeting_words:
                    return word
        return words[0]
    
    def _select_middle_word(self, words: List[str], prev_words: List[str], remaining_patterns: List[str]) -> str:
        """Select best middle word based on surrounding context."""
        if prev_words and prev_words[-1] in {'hi', 'hello', 'hey'}:
            # After greeting, prefer question words
            for word in words:
                if word in {'how', 'what', 'when', 'where'}:
                    return word
        return words[0]
    
    def _select_last_word(self, words: List[str], prev_words: List[str]) -> str:
        """Select best final word based on previous words."""
        if prev_words and prev_words[-1] in {'are', 'is'}:
            # After 'are/is', prefer 'you/the'
            for word in words:
                if word in {'you', 'the'}:
                    return word
        return words[0]
    
    def is_valid_pattern_sequence(self, pattern: str) -> bool:
        """Check if a pattern sequence could yield valid predictions."""
        # First check known patterns
        if pattern in self.common_phrases:
            return True
            
        return all(self.word_matcher.is_valid_pattern(p) 
                  for p in self.split_phrase_pattern(pattern))