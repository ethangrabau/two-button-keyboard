"""
Enhanced test suite for Two-Button Keyboard pattern matching system.
Tests performance, accuracy, and caching effectiveness.
"""

import sys
from pathlib import Path
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.prediction.pattern_matcher import PatternMatcher
from src.prediction.enhanced_phrase_matcher import EnhancedPhraseMatcher
from src.prediction.test_pattern_matcher import test_pattern_matching
from src.prediction.test_phrase_matcher import test_phrase_prediction, test_conversation_context

async def test_error_cases(pattern_matcher: PatternMatcher, phrase_matcher: EnhancedPhraseMatcher):
    """Test error handling and edge cases."""
    print("\nTest 4: Error Cases and Edge Cases")
    print("-" * 50)
    
    test_cases = [
        "",                     # Empty pattern
        "ABC",                  # Invalid characters
        "L" * 20,              # Very long pattern
        "R R R",               # Invalid spacing
        "LL RR RR LL " * 10    # Very long phrase
    ]
    
    for test in test_cases:
        print(f"\nTesting: '{test}'")
        
        # Test pattern matcher
        try:
            result = pattern_matcher.predict(test)
            print(f"Pattern matcher: {result}")
        except Exception as e:
            print(f"Pattern matcher error (expected): {str(e)}")
            
        # Test phrase matcher
        try:
            positions = await phrase_matcher.predict_phrase(test)
            if positions:
                words = [pos.candidates[0].word for pos in positions]
                print(f"Phrase matcher: {' '.join(words)}")
            else:
                print("Phrase matcher: No valid predictions")
        except Exception as e:
            print(f"Phrase matcher error (expected): {str(e)}")

async def run_tests():
    """Run complete test suite."""
    freq_file = str(project_root / "data" / "word_frequencies.json")
    pattern_matcher = PatternMatcher(freq_file)
    phrase_matcher = EnhancedPhraseMatcher(pattern_matcher)
    
    print("\nStarting Two-Button Keyboard Enhanced Test Suite")
    print("=" * 60)
    
    # Run tests
    pattern_results = await test_pattern_matching(pattern_matcher)
    phrase_results = await test_phrase_prediction(phrase_matcher)
    context_results = await test_conversation_context(phrase_matcher)
    await test_error_cases(pattern_matcher, phrase_matcher)
    
    # Print summary
    print("\nTest Summary")
    print("=" * 60)
    
    print("\nPattern Matching Results:")
    for key, value in pattern_results.get_stats().items():
        print(f"{key}: {value}")
        
    print("\nPhrase Prediction Results:")
    for key, value in phrase_results.get_stats().items():
        print(f"{key}: {value}")
        
    print("\nContext Awareness Results:")
    for key, value in context_results.get_stats().items():
        print(f"{key}: {value}")
    
    # Final system stats
    print("\nSystem Statistics:")
    pattern_stats = pattern_matcher.get_pattern_stats()
    phrase_stats = phrase_matcher.get_performance_stats()
    
    print("\nPattern Matcher:")
    for key, value in pattern_stats.items():
        print(f"  {key}: {value}")
        
    print("\nPhrase Matcher:")
    for cache_type, stats in phrase_stats.items():
        print(f"  {cache_type}:")
        for key, value in stats.items():
            print(f"    {key}: {value}")

if __name__ == "__main__":
    asyncio.run(run_tests())