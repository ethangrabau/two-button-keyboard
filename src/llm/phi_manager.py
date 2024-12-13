"""
Phi-2 LLM integration for Two-Button Keyboard.
"""

import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

class PhiManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.initialized = False
        self.init_lock = asyncio.Lock()

    async def initialize(self):
        """Initialize the Phi-2 model."""
        if self.initialized:
            return

        async with self.init_lock:
            if self.initialized:  # Double check in case of race
                return
                
            print("Initializing Phi-2 model...")
            start = time.time()
            
            # Load model and tokenizer
            model_id = "microsoft/phi-2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="mps"  # Use Metal for M3
            )
            
            duration = time.time() - start
            print(f"Model initialized in {duration:.1f}s")
            
            self.initialized = True

    async def generate_text(self, prompt: str, max_length: int = 50) -> str:
        """Generate text from prompt."""
        if not self.initialized:
            await self.initialize()
            
        # Encode prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        start = time.time()
        
        # Generate with low temperature for more focused predictions
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True
        )
        
        duration = time.time() - start
        print(f"Generated response in {duration:.3f}s")
        
        # Decode and clean response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        return response

    async def select_words(self, candidates: list[dict], context: str = "") -> list[str]:
        """Select best words from candidates."""
        prompt = f"""Context: {context}
Given these word candidates, select the most natural sequence:
{candidates}
Output only the selected words, space-separated:
"""
        response = await self.generate_text(prompt)
        return response.split()