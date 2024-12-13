"""
Test script for pattern-constrained model
"""

import torch
from transformers import AutoTokenizer
import logging
from model import PatternMatchConfig, PatternConstrainedModel
from keyboard_mapping import KeyboardMapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_pattern_constraints(device="cpu"):
    """Basic test of pattern constraints"""
    logging.info("Starting pattern constraint test")
    
    config = PatternMatchConfig(
        max_pattern_length=32,
        max_context_length=128,
        hidden_size=256,
        vocab_size=1000
    )
    
    # Initialize model
    model = PatternConstrainedModel(config).to(device)
    model.eval()
    
    # Create smaller test batch
    batch_size = 1
    seq_len = 8
    pattern_len = 4
    
    # Create input tensors with proper dtypes
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    pattern_ids = torch.randint(0, 3, (batch_size, pattern_len)).to(device)  # L=0, R=1, Space=2
    decoder_input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    
    # All masks as float tensors
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.float32).to(device)
    pattern_attention_mask = torch.ones((batch_size, pattern_len), dtype=torch.float32).to(device)
    decoder_attention_mask = torch.ones((batch_size, seq_len), dtype=torch.float32).to(device)
    
    # Pattern mask for vocabulary
    pattern_mask = torch.ones((batch_size, config.vocab_size), dtype=torch.bool).to(device)
    
    logging.info(f"Input shapes and types:")
    logging.info(f"- input_ids: {input_ids.shape}, dtype={input_ids.dtype}")
    logging.info(f"- pattern_ids: {pattern_ids.shape}, dtype={pattern_ids.dtype}")
    logging.info(f"- decoder_input_ids: {decoder_input_ids.shape}, dtype={decoder_input_ids.dtype}")
    logging.info(f"- attention_mask: {attention_mask.shape}, dtype={attention_mask.dtype}")
    
    # Test forward pass
    with torch.no_grad():
        try:
            logits = model(
                input_ids=input_ids,
                pattern_ids=pattern_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                pattern_mask=pattern_mask,
                pattern_attention_mask=pattern_attention_mask,
                decoder_attention_mask=decoder_attention_mask
            )
            logging.info(f"Output logits shape: {logits.shape}")
            assert logits.shape == (batch_size, seq_len, config.vocab_size)
            logging.info("Pattern constraint test passed!")
        except Exception as e:
            logging.error(f"Forward pass failed: {str(e)}")
            raise

def test_keyboard_integration():
    """Test integration between model and keyboard mapping"""
    logging.info("Starting keyboard integration test")
    
    mapper = KeyboardMapper()
    
    # Test words with known patterns - verified with character-by-character breakdown
    test_cases = [
        ("hello", "RLRRR"),      # h(R) e(L) l(R) l(R) o(R)
        ("thanks", "LRLRRL"),    # t(L) h(R) a(L) n(R) k(R) s(L)
        ("you", "RRR"),          # y(R) o(R) u(R)
        ("are", "LLL"),          # a(L) r(L) e(L)
        ("welcome", "LLRLRRL"),  # w(L) e(L) l(R) c(L) o(R) m(R) e(L)
        ("hi there", "RR LRLLL") # h(R) i(R) _(space) t(L) h(R) e(L) r(L) e(L)
    ]
    
    logging.info("Testing keyboard pattern mapping")
    for word, expected_pattern in test_cases:
        pattern = mapper.token_to_pattern(word)
        char_breakdown = " ".join(f"{c}({mapper.char_to_pattern(c)})" for c in word)
        assert pattern == expected_pattern, \
            f"Pattern mismatch for '{word}':\n" \
            f"- Got:      '{pattern}'\n" \
            f"- Expected: '{expected_pattern}'\n" \
            f"- Breakdown: {char_breakdown}"
        logging.info(f"✓ '{word}' -> '{pattern}' ({char_breakdown})")

def main():
    logging.info("Running all tests...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    try:
        test_keyboard_integration()  # Test keyboard first
        test_pattern_constraints(device)
        logging.info("All tests passed successfully!")
    except Exception as e:
        logging.error(f"Test failed with error: {e}")
        raise

if __name__ == "__main__":
    main()