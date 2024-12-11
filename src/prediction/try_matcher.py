from pattern_matcher import PatternMatcher
import os
from pathlib import Path

# Get the absolute path to word_frequencies.json
base_dir = Path(__file__).parent.parent
word_list_path = base_dir / "data" / "word_frequencies.json"

# Initialize the pattern matcher
matcher = PatternMatcher(str(word_list_path))

def test_prediction(pattern: str):
    """Test prediction for a given pattern."""
    print(f"\nTesting pattern: {pattern}")
    predictions = matcher.predict(pattern, max_results=3)
    print(f"Top predictions: {predictions}")
    return predictions

# Print some stats
print("Pattern Matcher Statistics:")
print(matcher.get_pattern_stats())

# Try some patterns
test_cases = [
    "L",      # Left side start
    "R",      # Right side start
    "LR",     # Two-letter pattern
    "LRL",    # Three-letter pattern
    "LRLL",   # Four-letter pattern
    "LRLLR"   # Five-letter pattern
]

for pattern in test_cases:
    test_prediction(pattern)

# Interactive testing
print("\nEnter patterns to test (q to quit)")
print("Use L for left side of keyboard, R for right side")
while True:
    pattern = input("\nEnter pattern (L/R): ").strip().upper()
    if pattern.lower() == 'q':
        break
    if not all(c in 'LR' for c in pattern):
        print("Please use only L and R characters")
        continue
    test_prediction(pattern)