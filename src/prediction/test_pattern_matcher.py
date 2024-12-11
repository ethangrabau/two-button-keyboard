"""
Tests for the PatternMatcher class.
"""

import unittest
import tempfile
import json
from pathlib import Path
from pattern_matcher import PatternMatcher

class TestPatternMatcher(unittest.TestCase):
    def setUp(self):
        """Create a temporary word list for testing."""
        self.test_words = {
            "test": 0.8,
            "hello": 0.9,
            "world": 0.7,
            "the": 1.0,
            "quick": 0.6
        }
        
        # Create temporary word list file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            json.dump(self.test_words, f)
            self.word_list_path = f.name
            
        self.matcher = PatternMatcher(self.word_list_path)
        
    def tearDown(self):
        """Clean up temporary files."""
        Path(self.word_list_path).unlink()
        
    def test_get_pattern(self):
        """Test pattern generation for words."""
        test_cases = [
            ("hello", "LRLLR"),
            ("world", "LRLLR"),
            ("test", "LRLL"),
            ("quick", "LLLRL")
        ]
        
        for word, expected in test_cases:
            with self.subTest(word=word):
                pattern = self.matcher.get_pattern(word)
                self.assertEqual(pattern, expected)
                
    def test_predict(self):
        """Test word prediction from patterns."""
        # Test complete pattern
        predictions = self.matcher.predict("LRLLR")
        self.assertIn("hello", predictions)  # Should return 'hello' before 'world' due to frequency
        
        # Test partial pattern
        predictions = self.matcher.predict("LR")
        self.assertGreater(len(predictions), 0)
        
    def test_is_valid_pattern(self):
        """Test pattern validation."""
        self.assertTrue(self.matcher.is_valid_pattern("LRLLR"))  # Valid complete pattern
        self.assertTrue(self.matcher.is_valid_pattern("LR"))     # Valid partial pattern
        self.assertFalse(self.matcher.is_valid_pattern("RRR"))   # Invalid pattern
        
    def test_get_word_frequency(self):
        """Test frequency lookup."""
        self.assertEqual(self.matcher.get_word_frequency("hello"), 0.9)
        self.assertEqual(self.matcher.get_word_frequency("nonexistent"), 0.0)
        
    def test_pattern_stats(self):
        """Test pattern statistics generation."""
        stats = self.matcher.get_pattern_stats()
        self.assertIn('total_patterns', stats)
        self.assertIn('total_words', stats)
        self.assertIn('avg_words_per_pattern', stats)
        self.assertIn('pattern_lengths', stats)
        
if __name__ == '__main__':
    unittest.main()