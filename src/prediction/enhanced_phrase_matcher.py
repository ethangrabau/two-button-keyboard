"""
Enhanced Phrase Pattern Matcher for Two-Button Keyboard Interface.
Provides structured word candidates for LLM-based selection.
"""

from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from pathlib import Path

@dataclass
class WordCandidate:
    word: str
    frequency_score: float
    pattern_confidence: float  # How well it matches the pattern
    context_score: float      # Based on surrounding words

@dataclass
class PositionCandidates:
    position: int
    pattern: str
    candidates: List[WordCandidate]
    
class EnhancedPhraseMatcher:
    def __init__(self, word_matcher):
        self.word_matcher = word_matcher
        self.context_weights = {
            'frequency': 0.4,
            'pattern': 0.3,
            'context': 0.3
        }
        
        # Load common phrase patterns
        self.load_common_phrases()
    
    def load_common_phrases(self):
        """Load common phrases and their patterns."""
        self.common_phrases = {
            "RR RRL LLL RRR": ["hi how are you"],
            "RLRRR RRL LLL RRR": ["hello how are you"],
            "RLR RRL LLL RRR": ["hey how are you"],
            "RRL LLL RRR LRRRL": ["how are you doing"]
        }
        
        # Analyze phrases to build context models
        self.context_pairs = {}  # word -> likely next words
        for phrases in self.common_phrases.values():
            for phrase in phrases:
                words = phrase.split()
                for i in range(len(words)-1):
                    if words[i] not in self.context_pairs:
                        self.context_pairs[words[i]] = {}
                    if words[i+1] not in self.context_pairs[words[i]]:
                        self.context_pairs[words[i]][words[i+1]] = 0
                    self.context_pairs[words[i]][words[i+1]] += 1
    
    def get_word_candidates(self, pattern: str, position: int, 
                          prev_words: List[str], max_candidates: int = 10) -> List[WordCandidate]:
        """Get ranked word candidates for a pattern at a specific position."""
        # Get base candidates from pattern matcher
        base_matches = self.word_matcher.predict(pattern, max_results=max_candidates*2)
        
        candidates = []
        for word in base_matches:
            # Calculate pattern confidence
            word_pattern = self.word_matcher.get_pattern(word)
            pattern_conf = 1.0 if word_pattern == pattern else 0.8
            
            # Get word frequency score
            freq_score = self.word_matcher.get_word_frequency(word)
            
            # Calculate context score
            context_score = self._calculate_context_score(word, position, prev_words)
            
            candidates.append(WordCandidate(
                word=word,
                frequency_score=freq_score,
                pattern_confidence=pattern_conf,
                context_score=context_score
            ))
        
        # Sort by combined score
        candidates.sort(key=lambda c: (
            c.frequency_score * self.context_weights['frequency'] +
            c.pattern_confidence * self.context_weights['pattern'] +
            c.context_score * self.context_weights['context']
        ), reverse=True)
        
        return candidates[:max_candidates]
    
    def _calculate_context_score(self, word: str, position: int, prev_words: List[str]) -> float:
        """Calculate how well a word fits in the current context."""
        if not prev_words:
            # First word scoring
            if position == 0:
                greeting_words = {'hi', 'hello', 'hey'}
                return 1.0 if word.lower() in greeting_words else 0.5
            return 0.5
            
        # Check if this is a common next word
        prev_word = prev_words[-1]
        if prev_word in self.context_pairs:
            total = sum(self.context_pairs[prev_word].values())
            score = self.context_pairs[prev_word].get(word, 0) / total
            if score > 0:
                return score
        
        # Position-based scoring
        if position > 0:
            if prev_word in {'hi', 'hello', 'hey'} and word in {'how', 'what', 'when'}:
                return 0.9
            elif prev_word in {'are', 'is'} and word in {'you', 'the', 'it'}:
                return 0.9
        
        return 0.5
    
    def predict_phrase(self, pattern_sequence: str, 
                      context: List[str] = None, 
                      max_candidates: int = 10) -> List[PositionCandidates]:
        """
        Generate structured word candidates for each position in a phrase pattern.
        Returns data suitable for LLM processing.
        """
        # First check common phrases
        if pattern_sequence in self.common_phrases:
            # Convert common phrase to structured format
            patterns = pattern_sequence.split()
            result = []
            for phrase in self.common_phrases[pattern_sequence]:
                words = phrase.split()
                for i, (word, pat) in enumerate(zip(words, patterns)):
                    result.append(PositionCandidates(
                        position=i,
                        pattern=pat,
                        candidates=[WordCandidate(
                            word=word,
                            frequency_score=1.0,
                            pattern_confidence=1.0,
                            context_score=1.0
                        )]
                    ))
            return result
            
        # Split into individual patterns
        patterns = pattern_sequence.split()
        if not patterns:
            return []
            
        # Generate candidates for each position
        prev_words = []
        result = []
        
        for i, pattern in enumerate(patterns):
            candidates = self.get_word_candidates(
                pattern=pattern,
                position=i,
                prev_words=prev_words,
                max_candidates=max_candidates
            )
            
            if not candidates:
                return []  # No valid candidates for this position
                
            result.append(PositionCandidates(
                position=i,
                pattern=pattern,
                candidates=candidates
            ))
            
            # Add best candidate to context for next iteration
            prev_words.append(candidates[0].word)
            
        return result
        
    def format_llm_prompt(self, positions: List[PositionCandidates], 
                         history: List[str] = None) -> str:
        """Format candidate positions for LLM processing."""
        prompt = "Select the most natural phrase using the following word candidates:\n\n"
        
        for pos in positions:
            candidates_str = ", ".join([
                f"{c.word} ({c.frequency_score:.2f})" 
                for c in pos.candidates
            ])
            prompt += f"Position {pos.position + 1} ({pos.pattern}): {candidates_str}\n"
        
        if history:
            prompt += "\nPrevious messages:\n"
            prompt += "\n".join(history[-3:])  # Last 3 messages for context
            
        prompt += "\nRequirements:\n"
        prompt += "- Select exactly one word from each position\n"
        prompt += "- Create a naturally flowing phrase\n"
        prompt += "- Consider conversation context\n"
        prompt += "- Maintain greeting/question patterns as appropriate\n\n"
        prompt += "Output format: word1 word2 word3 word4\n"
        
        return prompt