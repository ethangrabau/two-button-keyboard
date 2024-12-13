"""
Keyboard layout mapping for converting characters to L/R patterns.
- Each character maps to Left (L) or Right (R) based on QWERTY layout
- Space character maps to space ' ' in pattern
"""

class KeyboardMapper:
    # QWERTY keyboard layout split into left/right
    LEFT_CHARS = set('qwertasdfgzxcvb')
    RIGHT_CHARS = set('yuiophjklnm')
    
    def __init__(self):
        # Create case-insensitive mappings
        self.left_chars = self.LEFT_CHARS | {c.upper() for c in self.LEFT_CHARS}
        self.right_chars = self.RIGHT_CHARS | {c.upper() for c in self.RIGHT_CHARS}
        
    def char_to_pattern(self, char: str) -> str:
        """Convert a single character to L/R pattern or space."""
        if not char or len(char) != 1:
            return ''
            
        if char == ' ':
            return ' '  # Preserve space character
        elif char in self.left_chars:
            return 'L'
        elif char in self.right_chars:
            return 'R'
        else:
            return ''  # Invalid/unmapped character
            
    def token_to_pattern(self, token: str) -> str:
        """Convert a token to its L/R pattern, preserving spaces."""
        return ''.join(self.char_to_pattern(c) for c in token)
        
    def is_valid_pattern(self, pattern: str) -> bool:
        """Check if a pattern contains only valid L/R characters and spaces."""
        return all(c in {'L', 'R', ' '} for c in pattern)
        
    def token_matches_pattern(self, token: str, pattern: str) -> bool:
        """Check if a token matches the L/R pattern."""
        token_pattern = self.token_to_pattern(token)
        return token_pattern == pattern[:len(token_pattern)]

def print_keyboard_analysis():
    """Print the complete keyboard mapping analysis."""
    mapper = KeyboardMapper()
    test_words = [
        "hello",
        "thanks",
        "you",
        "are",
        "welcome",
        "hi there"
    ]
    print("\nKeyboard Pattern Analysis:")
    print("-" * 60)
    print(f"{'Word':<12} | {'Pattern':<12} | {'Character Breakdown'}")
    print("-" * 60)
    for word in test_words:
        pattern = mapper.token_to_pattern(word)
        breakdown = " ".join(f"{c}({mapper.char_to_pattern(c)})" for c in word)
        print(f"{word:<12} | {pattern:<12} | {breakdown}")
    print("-" * 60)

def test_keyboard_mapper():
    """Test the keyboard mapping functionality."""
    mapper = KeyboardMapper()
    print("\nRunning keyboard mapping tests...")
    
    # Print complete mapping analysis
    print_keyboard_analysis()
    
    # Test basic character mapping
    print("\n1. Testing basic character mapping...")
    assert mapper.char_to_pattern('q') == 'L', "q should be Left"
    assert mapper.char_to_pattern('p') == 'R', "p should be Right"
    assert mapper.char_to_pattern(' ') == ' ', "space should be space"
    print("✓ Basic character mapping passed")
    
    # Test token to pattern conversion
    print("\n2. Testing token to pattern conversion...")
    test_cases = [
        ("test", "LLLL"),
        ("hello", "RLRRR"),
        ("hi there", "RR LRLLL"),
        ("thanks", "LRLRRL"),
        ("you", "RRR"),
        ("are", "LLL"),  # Fixed this - all left side letters
        ("welcome", "LLRLRRL")
    ]
    
    for word, expected in test_cases:
        pattern = mapper.token_to_pattern(word)
        assert pattern == expected, f"Failed: '{word}' -> got '{pattern}', expected '{expected}'"
        print(f"✓ '{word}' correctly maps to '{pattern}'")
    
    print("\n3. Testing pattern validation...")
    assert mapper.is_valid_pattern('LLRR LRRL'), "Valid pattern failed"
    assert not mapper.is_valid_pattern('LLXR'), "Invalid pattern passed"
    print("✓ Pattern validation passed")
    
    print("\nAll keyboard mapping tests passed! 🎉")

if __name__ == "__main__":
    test_keyboard_mapper()