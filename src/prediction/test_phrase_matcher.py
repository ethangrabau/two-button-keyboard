"""
Test script for validating phrase pattern matching.
"""

from pattern_matcher import PatternMatcher
from phrase_matcher import PhraseMatcher
from pathlib import Path
import json

def create_test_words():
    """Create a test word list with common phrases."""
    return {
        # Greetings
        "hi": 1.0,
        "hello": 0.9,
        "hey": 0.95,
        
        # Question words
        "how": 0.99,
        "what": 0.98,
        "when": 0.97,
        "where": 0.96,
        
        # Common words
        "are": 0.95,
        "you": 0.94,
        "is": 0.93,
        "the": 0.92,
        "this": 0.91,
        "that": 0.90,
        
        # Test-specific words
        "today": 0.85,
        "doing": 0.84,
        "going": 0.83,
        "now": 0.82
    }

def test_phrases(phrase_matcher):
    """Test common phrase patterns."""
    test_cases = [
        # (pattern sequence, example phrases)
        ("RR RRR LLL RRR", ["hi how are you"]),
        ("RLRRR RRR LLL RRR", ["hello how are you"]),
        ("LRLL RL LRL", ["what is the"]),
        ("RRR LLL RRR LRLR", ["how are you doing"]),
    ]
    
    print("\nTesting phrase patterns:")
    print("-" * 50)
    
    for pattern, examples in test_cases:
        print(f"\nTesting pattern sequence: {pattern}")
        print(f"Example phrases: {examples}")
        
        predictions = phrase_matcher.predict_phrase(pattern)
        print(f"Predictions: {predictions}")
        
        # Verify predictions
        if predictions:
            phrase = predictions[0][0]  # Get top prediction
            valid = phrase_matcher.verify_phrase(phrase, pattern)
            print(f"Verification: {'✓' if valid else '❌'}")
            
            # Check if any example phrases were predicted
            if any(ex.lower() == phrase.lower() for ex in examples):
                print("✓ Found expected phrase")
            else:
                print("⚠️ Did not match expected phrases")
        else:
            print("❌ No predictions returned")
        
        # Test pattern generation
        if examples:
            generated = phrase_matcher.get_pattern_for_phrase(examples[0])
            print(f"Pattern generation test:")
            print(f"  Input phrase: {examples[0]}")
            print(f"  Generated pattern: {generated}")
            print(f"  Expected pattern: {pattern}")
            print(f"  Pattern match: {'✓' if generated == pattern else '❌'}")
        
        print("-" * 30)

def main():
    # Create test word list
    test_words = create_test_words()
    test_file = Path("test_words.json")
    
    with open(test_file, "w") as f:
        json.dump(test_words, f, indent=2)
    
    print("\nInitializing matchers...")
    word_matcher = PatternMatcher(str(test_file))
    phrase_matcher = PhraseMatcher(word_matcher)
    
    # Run tests
    test_phrases(phrase_matcher)
    
    # Cleanup
    test_file.unlink()

if __name__ == "__main__":
    main()