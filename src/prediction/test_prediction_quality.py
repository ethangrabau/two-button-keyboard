"""
Quality evaluation framework for Two-Button Keyboard predictions.
Tests prediction quality against real conversation data.
"""

import json
import logging
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class QualityTester:
    def __init__(self, word_frequencies_path: str = None):
        """Initialize tester with word frequencies for validation."""
        if word_frequencies_path is None:
            # Default to looking in parent directory's data folder
            current_dir = Path(__file__).parent.resolve()
            word_frequencies_path = current_dir.parent / 'data' / 'word_frequencies.json'
            
        self.word_frequencies = self._load_frequencies(str(word_frequencies_path))
        self.stats = {
            "perfect_matches": 0,
            "top_n_positions": [],
            "soft_matches": 0,
            "total_tests": 0
        }
        
    def _load_frequencies(self, path: str) -> Dict[str, float]:
        """Load word frequency dictionary."""
        try:
            logger.info(f"Loading frequencies from: {path}")
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading frequencies: {e}")
            raise
            
    def validate_prediction(self, prediction: str) -> bool:
        """Check if prediction only uses words from our frequency dictionary."""
        words = prediction.lower().split()
        return all(word in self.word_frequencies for word in words)
        
    def get_match_position(self, predictions: List[str], target: str) -> int:
        """Find position of exact match in predictions list."""
        try:
            return predictions.index(target) + 1
        except ValueError:
            return float('inf')
            
    def get_semantic_similarity(self, pred: str, target: str) -> float:
        """Calculate semantic similarity using BLEU score."""
        reference = [target.lower().split()]
        candidate = pred.lower().split()
        try:
            return sentence_bleu(reference, candidate)
        except:
            return 0.0
            
    def test_case(self, case: Dict[str, Any], predictor) -> Dict[str, Any]:
        """Test a single case and return results."""
        # Get predictions
        predictions = predictor.predict(
            context=case["context"],
            pattern=case["pattern"],
            max_results=5
        )
        
        # Validate predictions use known words
        valid_predictions = [p for p in predictions if self.validate_prediction(p)]
        
        if not valid_predictions:
            return {
                "success": False,
                "match_position": float('inf'),
                "semantic_score": 0.0,
                "error": "No valid predictions"
            }
            
        # Find exact match position
        match_pos = self.get_match_position(valid_predictions, case["target"])
        
        # Get semantic similarity of top prediction
        semantic_score = self.get_semantic_similarity(valid_predictions[0], case["target"])
        
        return {
            "success": True,
            "match_position": match_pos,
            "semantic_score": semantic_score,
            "predictions": valid_predictions[:5]
        }
        
    def test_all(self, test_cases: List[Dict[str, Any]], predictor) -> Dict[str, Any]:
        """Run all test cases and compile statistics."""
        self.stats = {
            "perfect_matches": 0,
            "top_n_positions": [],
            "soft_matches": 0,
            "total_tests": len(test_cases)
        }
        
        results = []
        for case in test_cases:
            result = self.test_case(case, predictor)
            results.append(result)
            
            if result["success"]:
                if result["match_position"] <= 5:
                    self.stats["top_n_positions"].append(result["match_position"])
                    if result["match_position"] == 1:
                        self.stats["perfect_matches"] += 1
                        
                if result["semantic_score"] > 0.5:  # Threshold for "soft" match
                    self.stats["soft_matches"] += 1
                    
        return {
            "stats": self.stats,
            "detailed_results": results
        }
        
    def print_results(self, results: Dict[str, Any]):
        """Print readable test results."""
        stats = results["stats"]
        print("\n=== Prediction Quality Test Results ===\n")
        print(f"Total Tests: {stats['total_tests']}")
        print(f"Perfect Matches: {stats['perfect_matches']} ({stats['perfect_matches']/stats['total_tests']*100:.1f}%)")
        
        if stats['top_n_positions']:
            print(f"Average N for matches: {np.mean(stats['top_n_positions']):.2f}")
            print(f"N distribution: {sorted(stats['top_n_positions'])}")
            
        print(f"Soft Matches: {stats['soft_matches']} ({stats['soft_matches']/stats['total_tests']*100:.1f}%)")
        
        print("\nDetailed Results:")
        for i, result in enumerate(results["detailed_results"]):
            print(f"\nTest {i+1}:")
            if result["success"]:
                print(f"  Match Position: {result['match_position']}")
                print(f"  Semantic Score: {result['semantic_score']:.3f}")
                print("  Top Predictions:")
                for j, pred in enumerate(result["predictions"][:3], 1):
                    print(f"    {j}. {pred}")
            else:
                print(f"  Error: {result['error']}")
                
if __name__ == "__main__":
    # Test cases using actual conversation data
    test_cases = [
        {
            "context": [
                "I'm headed up to mirror lake and will be up there most of tomorrow.",
                "Oh nice! Are they coming boating too?",
                "Sorry in rock canyon not sure if texts are going through or not",
                "They are!",
                "Ok sounds great! We are on our way to your place."
            ],
            "target": "Yeah I'll head on out",
            "pattern": "RLLR RRR RLLL RR RRL",
            "words": 5
        },
        {
            "context": [
                "Just emailed you what the ice cream shop place sent me in response to my email to them.",
                "Just read it! Dude that is awesome", 
                "That's awesome how profitable it is already.",
                "I'm really curious about what the franchise fees are",
                "Right! Looks great and yeah I'll send an email tomorrow asking him about that."
            ],
            "target": "Ok great thanks man I'll check it out",
            "pattern": "RR LLLL LRLRRL RLR RRR LRLLR RL RRL",
            "words": 8
        }
    ]
    
    # Initialize tester
    tester = QualityTester()
    
    # Get predictor
    from bert_predictor import BertPredictor
    predictor = BertPredictor()
    
    # Run tests
    results = tester.test_all(test_cases, predictor)
    
    # Print results
    tester.print_results(results)