"""
Interface for Phi-2 model using transformers library.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
import logging
import functools
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cache_prediction(func):
    """Cache predictions for 5 minutes."""
    cache = {}
    def clear_old_entries():
        now = datetime.now()
        expired = [k for k, (_, t) in cache.items() if now - t > timedelta(minutes=5)]
        for k in expired:
            del cache[k]
            
    def wrapper(*args, **kwargs):
        clear_old_entries()
        # Create cache key from relevant arguments
        key = str(args[1:]) + str(sorted(kwargs.items()))
        if key in cache:
            result, _ = cache[key]
            return result
        result = func(*args, **kwargs)
        cache[key] = (result, datetime.now())
        return result
    return wrapper

class Phi2Interface:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.tokenizer = None
        
    async def initialize(self):
        """Initialize Phi-2 model and tokenizer."""
        try:
            logger.info("Loading Phi-2 model and tokenizer...")
            model_name = "microsoft/phi-2"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                low_cpu_mem_usage=True
            )
            
            logger.info("Phi-2 model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load Phi-2: {e}")
            return False
            
    def format_candidates_prompt(self, positions: List[Dict], message_history: List[str] = None) -> str:
        """Format word candidates into a prompt for Phi-2."""
        prompt = "Task: Select words to form a natural phrase from the given choices.\n\n"
        
        # Add context if available
        if message_history:
            last_msg = message_history[-1] if message_history else ""
            prompt += f"Context: Responding to '{last_msg}'\n\n"
        
        # Add word options
        prompt += "Available words for each position:\n"
        for i, pos in enumerate(positions):
            candidates = sorted(pos.candidates, key=lambda x: x.frequency_score, reverse=True)[:3]
            options = [c.word for c in candidates]
            prompt += f"{i+1}: {', '.join(options)}\n"
        
        prompt += "\nSelected words:"
        return prompt
        
    @cache_prediction
    def select_words(self, 
                    positions: List[Dict],
                    message_history: List[str] = None) -> Tuple[List[str], float]:
        """Select words using Phi-2."""
        if not self.model or not self.tokenizer:
            logger.warning("Phi-2 not initialized, falling back to pattern matching")
            return [pos.candidates[0].word for pos in positions], 0.0
            
        try:
            # For single words, use pattern matching
            if len(positions) == 1:
                word = positions[0].candidates[0].word
                confidence = positions[0].candidates[0].frequency_score
                return [word], confidence
            
            # For phrases, use LLM
            prompt = self.format_candidates_prompt(positions, message_history)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=32,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_beams=1,
                    do_sample=False
                )
                
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            selected_text = generated.split("Selected words:")[-1].strip()
            selected_words = selected_text.split()[:len(positions)]
            
            # Validate selection
            valid_words = []
            confidence = 0.0
            
            for i, word in enumerate(selected_words):
                word = word.lower().strip('.,!?')
                pos = positions[i]
                candidates = {c.word.lower(): c for c in pos.candidates}
                
                if word in candidates:
                    valid_words.append(candidates[word].word)
                    confidence += candidates[word].frequency_score
                else:
                    # Fall back to best frequency match
                    valid_words.append(pos.candidates[0].word)
                    confidence += 0.5
            
            # Fill in any missing positions
            while len(valid_words) < len(positions):
                pos = positions[len(valid_words)]
                valid_words.append(pos.candidates[0].word)
                confidence += 0.5
            
            confidence = confidence / len(positions)
            
            logger.info(f"Selected: {' '.join(valid_words)} (conf: {confidence:.2f})")
            return valid_words, confidence
            
        except Exception as e:
            logger.error(f"Error in word selection: {e}")
            return [pos.candidates[0].word for pos in positions], 0.0
            
    def cleanup(self):
        """Clean up model resources."""
        try:
            if hasattr(self.model, 'to'):
                self.model.to('cpu')
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
