"""
Test script for validating pattern matcher predictions.
"""

from pattern_matcher import PatternMatcher
from pathlib import Path
import json

def create_test_words():
    """Create a test word list with common words and their frequencies."""
    return {
        # Single letter words
        "a": 1.0,
        "i": 0.99,
        
        # Two letter words
        "hi": 0.95,
        "in": 0.94,
        "is": 0.93,
        "it": 0.92,
        "to": 0.91,
        "no": 0.90,
        "up": 0.89,
        
        # Three letter words
        "the": 0.88,
        "and": 0.87,
        "how": 0.86,
        "now": 0.85,
        "was": 0.84,
        "for": 0.83,
        
        # Four letter words
        "this": 0.82,
        "that": 0.81,
        "with": 0.80,
        "have": 0.79,
        "test": 0.78,
        
        # Five+ letter words
        "hello": 0.77,
        "there": 0.76,
        "world": 0.75,
        "should": 0.74,
        "testing": 0.73,
        "working": 0.72
    }

def verify_pattern(matcher, word, expected_pattern):
    """Verify that a word maps to the expected pattern."""
    actual_pattern = matcher.get_pattern(word)
    if actual_pattern != expected_pattern:
        print(f"⚠️  Pattern mismatch for '{word}':")
        print(f"   Expected: {expected_pattern}")
        print(f"   Got:      {actual_pattern}")
        return False
    return True

def run_test_cases(matcher):
    """Run through a series of test cases."""
    # First verify some known patterns
    print("\nVerifying word patterns:")
    verify_cases = [
        ("hi", "RR"),
        ("how", "RRL"),
        ("the", "LRL"),
        ("test", "LLLL"),
        ("hello", "RLRRR"),
    ]
    
    for word, expected in verify_cases:
        verify_pattern(matcher, word, expected)
    
    # Test predictions
    test_cases = [
        # Pattern, expected length, example words that should match
        ("R", 1, ["i", "a"]),
        ("L", 1, []),  # No single-letter words on left side
        ("RR", 2, ["hi", "it"]),
        ("RL", 2, ["to"]),
        ("RRL", 3, ["how", "now"]),
        ("LRL", 3, ["the"]),
        ("LLLL", 4, ["test"]),
        ("RLRRR", 5, ["hello"]),
    ]
    
    print("\nRunning prediction tests:")
    print("-" * 50)
    
    for pattern, expected_len, example_words in test_cases:
        print(f"\nTesting pattern: {pattern}")
        print(f"Expected length: {expected_len}")
        print(f"Example words that should match: {example_words}")
        
        predictions = matcher.predict(pattern, max_results=3)
        print(f"Got predictions: {predictions}")
        
        # Validate predictions
        if predictions:
            actual_pattern = matcher.get_pattern(predictions[0])
            print(f"Top prediction pattern: {actual_pattern}")
            if len(actual_pattern) != expected_len:
                print(f"⚠️ Warning: Expected length {expected_len}, but got {len(actual_pattern)}")
            
            # Check if any example words appear in predictions
            found = [w for w in example_words if w in predictions]
            if found:
                print(f"✓ Found expected words: {found}")
            elif example_words:
                print(f"⚠️ Did not find any expected words: {example_words}")
        else:
            if example_words:
                print("❌ No predictions returned but expected some matches")
            else:
                print("✓ Correctly returned no predictions")
            
        print("-" * 30)

def main():
    # Create a test word list
    test_words = create_test_words()
    test_file = Path("test_words.json")
    
    print("Creating test word list...")
    with open(test_file, "w") as f:
        json.dump(test_words, f, indent=2)
    
    print("\nInitializing pattern matcher...")
    matcher = PatternMatcher(str(test_file))
    
    # Print initial stats
    stats = matcher.get_pattern_stats()
    print("\nPattern matcher stats:")
    for key, value in stats.items():
        if key != 'pattern_samples':
            print(f"{key}: {value}")
    
    # Run test cases
    run_test_cases(matcher)
    
    # Cleanup
    test_file.unlink()

if __name__ == "__main__":
    main()