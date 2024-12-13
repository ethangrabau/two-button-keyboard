"""
Test pattern masking functionality
"""

import torch
from transformers import GPT2TokenizerFast
from model_common import PatternMask
from keyboard_mapping import KeyboardMapper
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_pattern_masking():
    """Test the pattern masking system"""
    logger.info("Testing pattern masking...")
    
    # Initialize components
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    pattern_mask = PatternMask(tokenizer)
    keyboard = KeyboardMapper()
    
    # Test with some sample patterns
    test_patterns = [
        "LLRR",      # Test basic pattern
        "L R L",     # Test with spaces
        "LLRRLL",    # Test longer pattern
        "LR"         # Test short pattern
    ]
    
    # Convert patterns to IDs
    pattern_map = {'L': 0, 'R': 1, ' ': 2}
    for test_pattern in test_patterns:
        logger.info(f"\nTesting pattern: {test_pattern}")
        
        # Convert to tensor
        pattern_ids = []
        for char in test_pattern:
            if char in pattern_map:
                pattern_ids.append(pattern_map[char])
        while len(pattern_ids) < 32:  # Pad to max length
            pattern_ids.append(3)  # Padding ID
            
        pattern_tensor = torch.tensor([pattern_ids])
        
        # Create masks
        masks = pattern_mask.create_pattern_masks(pattern_tensor)
        
        # Print stats
        allowed_tokens = masks[0, 0].sum().item()
        logger.info(f"Total tokens allowed: {allowed_tokens}")
        
        # Test some known words
        test_words = ["hello", "test", "the", "and", "you"]
        for word in test_words:
            # Get word's pattern
            word_pattern = keyboard.token_to_pattern(word)
            if not word_pattern:
                continue
                
            # Check if word matches our test pattern
            token_ids = tokenizer.encode(word)
            for tid in token_ids:
                if masks[0, 0, tid]:
                    logger.info(f"Word '{word}' (pattern: {word_pattern}) is allowed")
                    
        # Validate special tokens
        special_tokens = {
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id
        }
        for token_id in special_tokens:
            assert masks[0, 0, token_id] == 1, f"Special token {token_id} should be allowed"
            
    logger.info("\nAll pattern mask tests passed! 🎉")

if __name__ == "__main__":
    test_pattern_masking()