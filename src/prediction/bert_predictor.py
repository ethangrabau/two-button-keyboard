"""
BERT-based predictor for Two-Button Keyboard.
Takes context and L/R patterns to predict next message.
"""

import json
import logging
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM
import torch
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class BertPredictor:
    def __init__(self):
        """Initialize BERT model and tokenizer."""
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load BERT
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(self.device)
        
        # Load word frequencies with proper path resolution
        current_dir = Path(__file__).parent.resolve()
        data_dir = current_dir.parent / 'data'
        freq_path = data_dir / 'word_frequencies.json'
        logger.info(f"Looking for word frequencies at: {freq_path}")
        
        try:
            with open(freq_path, 'r') as f:
                self.word_frequencies = json.load(f)
                logger.info(f"Loaded {len(self.word_frequencies)} words")
        except Exception as e:
            logger.error(f"Error loading word frequencies: {e}")
            raise
            
        # Build pattern lookup
        self.pattern_to_words = {}
        self._build_pattern_lookup()
        
    def _build_pattern_lookup(self):
        """Build dictionary mapping L/R patterns to possible words."""
        left_keys = set('qwertasdfgzxcv')
        for word in self.word_frequencies:
            pattern = ''.join('L' if c.lower() in left_keys else 'R' 
                            for c in word)
            if pattern not in self.pattern_to_words:
                self.pattern_to_words[pattern] = []
            self.pattern_to_words[pattern].append(word)
            
        logger.info(f"Built pattern lookup with {len(self.pattern_to_words)} patterns")
            
    def _get_word_candidates(self, pattern: str) -> List[str]:
        """Get list of words matching an L/R pattern."""
        candidates = self.pattern_to_words.get(pattern, [])
        if not candidates:
            logger.warning(f"No candidates found for pattern: {pattern}")
        # Sort by frequency and take top 10 candidates to limit combinations
        candidates = sorted(candidates, 
                          key=lambda w: self.word_frequencies.get(w, 0), 
                          reverse=True)[:10]
        return candidates
        
    def _score_sequence(self, context: str, candidate_sequence: str) -> float:
        """Score a complete candidate sequence using BERT."""
        # Prepare input: [CLS] context [SEP] candidate [SEP]
        input_text = f"{context} [SEP] {candidate_sequence}"
        inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get BERT prediction scores for the complete sequence
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Get sequence-level score
            logits = outputs.logits[0]
            sequence_tokens = self.tokenizer.encode(candidate_sequence, add_special_tokens=False)
            score = 0
            
            # Score each token in the sequence against BERT's predictions
            for i, token_id in enumerate(sequence_tokens):
                token_scores = logits[i].softmax(dim=0)
                score += token_scores[token_id].item()
                
            return score / len(sequence_tokens)  # Normalize by sequence length
        
    def predict(self, context: List[str], pattern: str, max_results: int = 5) -> List[str]:
        """
        Predict next message given context and L/R pattern.
        
        Args:
            context: List of previous messages
            pattern: Space-separated L/R patterns for each word
            max_results: Maximum number of predictions to return
            
        Returns:
            List of predicted messages, sorted by probability
        """
        # Split pattern into words
        word_patterns = pattern.split()
        logger.info(f"Predicting for pattern: {pattern} ({len(word_patterns)} words)")
        
        # Get candidates for each word position
        word_candidates = []
        total_combinations = 1
        for pat in word_patterns:
            candidates = self._get_word_candidates(pat)
            if not candidates:
                logger.warning(f"No candidates found for pattern: {pat}")
                candidates = ["<unknown>"]  # Placeholder for unknown patterns
            word_candidates.append(candidates)
            total_combinations *= len(candidates)
            logger.info(f"Found {len(candidates)} candidates for pattern {pat}")
        
        logger.info(f"Total possible combinations: {total_combinations}")
        if total_combinations > 10000:
            logger.warning("Too many combinations, limiting candidates further")
            # Take top 5 candidates for each position
            word_candidates = [sorted(candidates, 
                                   key=lambda w: self.word_frequencies.get(w, 0), 
                                   reverse=True)[:5] 
                             for candidates in word_candidates]
            total_combinations = 1
            for candidates in word_candidates:
                total_combinations *= len(candidates)
            logger.info(f"Reduced to {total_combinations} combinations")
            
        # Generate candidate sequences (complete messages)
        sequences = []
        context_str = " [SEP] ".join(context[-3:])  # Use last 3 messages
        
        # Try combinations of candidates (complete sequences only)
        import itertools
        for i, words in enumerate(itertools.product(*word_candidates)):
            if i % 100 == 0:  # Log progress
                logger.info(f"Processing combination {i+1}/{total_combinations}")
            candidate_seq = " ".join(words)
            score = self._score_sequence(context_str, candidate_seq)
            sequences.append((candidate_seq, score))
            
        # Sort by score and return top sequences
        sequences.sort(key=lambda x: x[1], reverse=True)
        return [seq for seq, _ in sequences[:max_results]]